#!/usr/bin/env python3
"""Collect public brand images only after visible brand-product verification.

Pipeline: last two years of public posts -> newest-first candidates ->
Qwen3.5-9B product/logo check -> OCR cross-check -> visual diversity selection.
Unverified images are not selected.

Install:
    python -m pip install instaloader pillow numpy scipy requests pytesseract
Also install the Tesseract executable. Serve Qwen3.5-9B through an
OpenAI-compatible endpoint before running this script.

Example:
    python instagram_brand_image_collector.py --images-per-brand 20 \
      --vlm-base-url http://localhost:8000/v1 \
      --vlm-model Qwen/Qwen3.5-9B

Maison Margiela and Celine with Ollama:
    python instagram_brand_image_collector.py \
      --brands "Maison Margiela" Celine \
      --images-per-brand 15 \
      --scan-posts 600 \
      --candidate-posts 30 \
      --vlm-base-url http://localhost:11434/v1 \
      --vlm-model qwen3.5:9b

Only public profile posts are accessed. No login, comments, captions, geotags,
stories, private content, or access-control bypass is used.
"""

import argparse
import base64
import csv
import io
import json
import os
import re
import shutil
import time
from datetime import datetime, timedelta, timezone
from itertools import islice
from pathlib import Path
from urllib.parse import urlparse

import instaloader
import numpy as np
import pytesseract
import requests
from PIL import Image, ImageOps
from scipy.fftpack import dct


BRAND_PROFILES = {
    "Maison Margiela": "https://www.instagram.com/maisonmargiela/",
    "Marshall": "https://www.instagram.com/marshall/",
    "Balenciaga": "https://www.instagram.com/balenciaga/",
    "Celine": "https://www.instagram.com/celine/",
    "Lacoste": "https://www.instagram.com/lacoste/",
}

BRAND_ALIASES = {
    "Maison Margiela": ("maison margiela", "maisonmargiela", "margiela"),
    "Marshall": ("marshall",),
    "Balenciaga": ("balenciaga",),
    "Celine": ("celine",),
    "Lacoste": ("lacoste",),
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TEXT_ANYWHERE_BRANDS = {"Maison Margiela", "Celine"}
MANIFEST_FIELDS = [
    "brand",
    "account",
    "account_url",
    "post_shortcode",
    "post_date_utc",
    "post_url",
    "post_type",
    "month",
    "selected_image",
    "candidate_image",
    "phash",
    "verification_status",
    "detected_products",
    "evidence_type",
    "evidence_location",
    "evidence_description",
    "vlm_confidence",
    "ocr_brand_match",
    "review_reason",
    "verification_model",
]


def username_from_url(account_url):
    parsed = urlparse(account_url)
    if (parsed.hostname or "").lower() not in {"instagram.com", "www.instagram.com"}:
        raise ValueError(f"Not an Instagram URL: {account_url}")
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 1 or not re.fullmatch(r"[A-Za-z0-9._]+", parts[0]):
        raise ValueError(f"Not a profile URL: {account_url}")
    if parts[0].lower() in {"p", "reel", "reels", "explore", "stories"}:
        raise ValueError(f"Not a profile URL: {account_url}")
    return parts[0]


def parse_date(value):
    if not value:
        return None
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def spread_posts_by_date(posts, limit):
    """Select evenly spaced posts from oldest to newest, preserving date range."""
    posts = sorted(posts, key=lambda post: post.date_utc)
    if len(posts) <= limit:
        return posts
    indices = np.linspace(0, len(posts) - 1, num=limit, dtype=int)
    return [posts[index] for index in dict.fromkeys(indices.tolist())]


def perceptual_hash(image):
    """Return a 64-bit perceptual hash for coarse pattern/structure diversity."""
    pixels = np.asarray(
        image.convert("L").resize((32, 32), Image.Resampling.LANCZOS),
        dtype=float,
    )
    coefficients = dct(dct(pixels, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    median = np.median(coefficients.flatten()[1:])
    return (coefficients > median).flatten()


def colour_histogram(image):
    hsv = np.asarray(image.convert("HSV").resize((128, 128), Image.Resampling.BILINEAR))
    parts = []
    for channel, bins, value_range in (
        (0, 8, (0, 256)),
        (1, 4, (0, 256)),
        (2, 4, (0, 256)),
    ):
        histogram, _ = np.histogram(hsv[:, :, channel], bins=bins, range=value_range)
        parts.extend((histogram / max(histogram.sum(), 1)).tolist())
    return np.asarray(parts)


def add_visual_features(record):
    with Image.open(record["candidate_path"]) as image:
        record["hash"] = perceptual_hash(image)
        record["colour"] = colour_histogram(image)
    record["phash"] = hex(int("".join("1" if bit else "0" for bit in record["hash"]), 2))
    return record


def normalise_text(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def ocr_brand_match(image_path, brand):
    """Use OCR only as supporting evidence for text physically visible in an image."""
    with Image.open(image_path) as image:
        text = pytesseract.image_to_string(ImageOps.exif_transpose(image))
    normalised = normalise_text(text)
    return any(normalise_text(alias) in normalised for alias in BRAND_ALIASES[brand])


def image_data_url(image_path):
    """Downscale and encode an image for an OpenAI-compatible vision request."""
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def parse_json_object(text):
    """Accept plain or fenced JSON, including literal controls in model strings."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"VLM did not return a JSON object: {text[:300]}")
    return json.loads(cleaned[start:end + 1], strict=False)


def qwen_brand_check(image_path, brand, args):
    evidence_rule = (
        f"For {brand}, the brand name or logo may appear anywhere in the image, "
        "including a caption, watermark, social-media overlay, or sign."
        if brand in TEXT_ANYWHERE_BRANDS else
        "The brand name or logo must be physically visible on the product, its "
        "sewn/printed label, or its packaging. A caption, watermark, social-media "
        "overlay, storefront sign, or person's name is not product evidence."
    )
    prompt = f"""Audit this image for the target brand: {brand}.
Do not infer the brand from the Instagram account, clothing style, product shape,
colours, or a general aesthetic. Identify products and visible evidence only.

{evidence_rule}
A physical product must actually be visible in the image. Books, zines, documents,
text pages, illustrations, drawings, signs, and text describing a product do not
count as physical products. Clothing or objects depicted within an illustration,
book cover, or artwork also do not count. Set physical_product_present to false
for these cases, and return an empty products list.
For Maison Margiela, numeric markings or four stitches alone are not sufficient.
Return review whenever visibility or identity is unclear.

Return only one JSON object with these fields:
{{
  "products": ["visible physical product 1", "visible physical product 2"],
  "physical_product_present": true,
  "clothing_or_accessory_present": true,
  "brand_text_on_product": true,
  "brand_logo_on_product": true,
  "evidence_location": "precise visible location, or none",
  "evidence_description": "what text/logo is visible and on which product",
  "confidence": "high",
  "recommendation": "verify"
}}
Allowed confidence values: high, medium, low.
Allowed recommendation values: verify, review, reject.
Use verify only for clear visible brand evidence satisfying the rule above.
"""
    payload = {
        "model": args.vlm_model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
            ],
        }],
        "temperature": 0,
        "max_tokens": 1200,
        "stream": False,
    }
    parsed_endpoint = urlparse(args.vlm_base_url)
    use_ollama = args.vlm_provider == "ollama" or (
        args.vlm_provider == "auto" and parsed_endpoint.port == 11434
    )
    if use_ollama:
        payload["reasoning_effort"] = "none"
    else:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    headers = {"Content-Type": "application/json"}
    if args.vlm_api_key:
        headers["Authorization"] = f"Bearer {args.vlm_api_key}"
    response = requests.post(
        f"{args.vlm_base_url.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=args.vlm_timeout,
    )
    response.raise_for_status()
    body = response.json()
    choice = body["choices"][0]
    message = choice["message"]
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        )
    if not content.strip():
        reasoning = (
            message.get("reasoning") or message.get("reasoning_content")
            or message.get("thinking") or ""
        )
        raise ValueError(
            "VLM returned empty content "
            f"(finish_reason={choice.get('finish_reason')!r}, "
            f"reasoning_characters={len(reasoning)}). "
            "Check that thinking is disabled and the response token limit is sufficient."
        )
    return parse_json_object(content)


def as_bool(value):
    return value is True or str(value).strip().lower() == "true"


def verify_brand_product(record, brand, args):
    """Apply deterministic acceptance rules after the VLM assessment."""
    result = qwen_brand_check(record["candidate_path"], brand, args)
    products = result.get("products") or []
    if isinstance(products, str):
        products = [products]
    physical_product = as_bool(result.get("physical_product_present"))
    text_visible = as_bool(result.get("brand_text_on_product"))
    logo_visible = as_bool(result.get("brand_logo_on_product"))
    confidence = str(result.get("confidence", "low")).lower()
    recommendation = str(result.get("recommendation", "reject")).lower()
    location = str(result.get("evidence_location", "")).strip()
    description = str(result.get("evidence_description", "")).strip()
    location_known = normalise_text(location) not in {"", "none", "unknown", "na", "notvisible"}
    ocr_match = ocr_brand_match(record["candidate_path"], brand)

    if brand in TEXT_ANYWHERE_BRANDS and physical_product and products and (
        ocr_match or (recommendation == "verify" and (text_visible or logo_visible))
    ):
        status, reason = "verified", "Brand text or logo and a physical product were detected"
    elif brand in TEXT_ANYWHERE_BRANDS and not physical_product:
        status, reason = "rejected", "No actual physical product was visible"
    elif not products:
        status, reason = "rejected", "No identifiable product was reported"
    elif recommendation == "reject":
        status, reason = "rejected", "VLM rejected the brand-product relationship"
    elif not (text_visible or logo_visible):
        status, reason = "rejected", "No visible on-product brand text or logo"
    elif recommendation != "verify":
        status, reason = "review", "VLM did not recommend verification"
    elif confidence != "high":
        status, reason = "review", "Visible evidence was not high confidence"
    elif not location_known:
        status, reason = "review", "Evidence location was not specified"
    elif not description:
        status, reason = "review", "Visible evidence was not described"
    elif text_visible and not logo_visible and not ocr_match:
        status, reason = "review", "VLM text claim lacked independent OCR support"
    else:
        status, reason = "verified", "Visible on-product brand evidence passed all checks"

    evidence_type = "both" if text_visible and logo_visible else (
        "text" if text_visible else "logo" if logo_visible else "none"
    )
    return {
        "verification_status": status,
        "detected_products": "; ".join(str(item) for item in products),
        "evidence_type": evidence_type,
        "evidence_location": location,
        "evidence_description": description,
        "vlm_confidence": confidence,
        "ocr_brand_match": ocr_match,
        "review_reason": reason,
        "verification_model": args.vlm_model,
    }


def visual_distance(left, right):
    hash_distance = np.count_nonzero(left["hash"] != right["hash"]) / 64.0
    colour_distance = 0.5 * np.abs(left["colour"] - right["colour"]).sum()
    return 0.7 * hash_distance + 0.3 * colour_distance


def select_diverse_images(records, target):
    """Cover different months first, then greedily maximise visual diversity."""
    if len(records) <= target:
        return records

    records = sorted(records, key=lambda record: record["date_utc"])
    months = sorted({record["date_utc"][:7] for record in records})
    if len(months) > target:
        month_indices = np.linspace(0, len(months) - 1, num=target, dtype=int)
        months_to_cover = [months[index] for index in dict.fromkeys(month_indices.tolist())]
    else:
        months_to_cover = months

    selected = []
    selected_paths = set()
    selected_shortcodes = set()

    def best_candidate(candidates):
        candidates = [
            record for record in candidates
            if record["post_shortcode"] not in selected_shortcodes
        ]
        if not candidates:
            return None
        if not selected:
            return candidates[len(candidates) // 2]
        return max(
            candidates,
            key=lambda record: min(visual_distance(record, other) for other in selected),
        )

    # Phase 1: one image from each covered month where possible.
    for month in months_to_cover:
        choice = best_candidate([r for r in records if r["date_utc"][:7] == month])
        if choice is not None:
            selected.append(choice)
            selected_paths.add(choice["candidate_path"])
            selected_shortcodes.add(choice["post_shortcode"])

    # Phase 2: fill remaining slots with visually distant images.
    while len(selected) < target:
        candidates = [
            record for record in records
            if record["candidate_path"] not in selected_paths
            and record["post_shortcode"] not in selected_shortcodes
        ]
        if not candidates:
            break
        selected_months = {record["date_utc"][:7] for record in selected}
        choice = max(
            candidates,
            key=lambda record: (
                min(visual_distance(record, other) for other in selected)
                + (0.15 if record["date_utc"][:7] not in selected_months else 0.0)
            ),
        )
        selected.append(choice)
        selected_paths.add(choice["candidate_path"])
        selected_shortcodes.add(choice["post_shortcode"])

    return sorted(selected, key=lambda record: record["date_utc"])


def image_files(directory):
    return {
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def make_loader(output_dir, delay):
    return instaloader.Instaloader(
        dirname_pattern=str(output_dir / "candidates" / "{target}"),
        filename_pattern="{date_utc:%Y%m%dT%H%M%SZ}_{shortcode}",
        download_pictures=True,
        download_videos=False,
        download_video_thumbnails=False,
        download_comments=False,
        download_geotags=False,
        save_metadata=False,
        post_metadata_txt_pattern="",
        request_timeout=30,
        max_connection_attempts=2,
        quiet=False,
    )


def collect_brand(brand, account_url, args, loader):
    username = username_from_url(account_url)
    profile = instaloader.Profile.from_username(loader.context, username)
    if profile.is_private:
        raise RuntimeError(f"{brand}: private profiles are not supported")

    since = parse_date(args.since) or (datetime.now(timezone.utc) - timedelta(days=730))
    until = parse_date(args.until)
    eligible_posts = []
    for post in islice(profile.get_posts(), args.scan_posts):
        if post.is_video:
            continue
        post_date = post.date_utc
        if post_date.tzinfo is None:
            post_date = post_date.replace(tzinfo=timezone.utc)
        if since and post_date < since:
            continue
        if until and post_date > until:
            continue
        eligible_posts.append(post)

    eligible_posts.sort(key=lambda post: post.date_utc, reverse=True)
    candidate_posts = eligible_posts[:args.candidate_posts]
    if not candidate_posts:
        raise RuntimeError(f"{brand}: no eligible public image posts found")

    print(f"{brand}: {len(eligible_posts)} eligible posts from the last two years; "
          f"checking up to {len(candidate_posts)} most recent candidates")
    records = []
    candidate_dir = args.output / "candidates" / brand
    downloaded_posts = []

    def download_records(post):
        before = image_files(candidate_dir) if candidate_dir.exists() else set()
        loader.download_post(post, target=brand)
        downloaded_posts.append(post)
        new_files = sorted(image_files(candidate_dir) - before)
        if not new_files:  # Supports reruns when files already exist.
            new_files = sorted(
                path for path in image_files(candidate_dir)
                if post.shortcode in path.stem
            )
        if not new_files:
            print(f"{brand}: no image saved for {post.shortcode}; skipping")
            return []

        post_records = []
        # Inspect every carousel slide: a label/logo may appear only in a later
        # slide. The final diversity selector still permits at most one image
        # per post, so a carousel cannot dominate the selected set.
        for path in new_files:
            record = {
                "brand": brand,
                "account": username,
                "account_url": account_url,
                "post_shortcode": post.shortcode,
                "post_date_utc": post.date_utc.isoformat(),
                "date_utc": post.date_utc.isoformat(),
                "post_url": f"https://www.instagram.com/p/{post.shortcode}/",
                "post_type": post.typename,
                "candidate_path": path,
            }
            try:
                post_records.append(add_visual_features(record))
            except (OSError, ValueError) as exc:
                print(f"{brand}: unreadable image {path.name}: {exc}")
        time.sleep(args.delay)
        return post_records

    def verify_records(new_records):
        for record in new_records:
            if sum(r["verification_status"] == "verified" for r in records) >= args.images_per_brand:
                break
            print(f"{brand}: verifying product evidence {len(records) + 1}")
            record.update(verify_brand_product(record, brand, args))
            records.append(record)

    for post in candidate_posts:
        if sum(r["verification_status"] == "verified" for r in records) >= args.images_per_brand:
            break
        verify_records(download_records(post))

    if not records:
        raise RuntimeError(f"{brand}: no readable candidate images downloaded")

    attempted_shortcodes = {post.shortcode for post in downloaded_posts}
    additional_posts = [
        post for post in eligible_posts if post.shortcode not in attempted_shortcodes
    ]
    for post in additional_posts:
        verified_count = sum(r["verification_status"] == "verified" for r in records)
        if verified_count >= args.images_per_brand:
            break
        print(f"{brand}: {verified_count}/{args.images_per_brand} verified; "
              f"trying additional post {post.shortcode}")
        verify_records(download_records(post))

    # Brand-product relevance is decided before date/visual diversity selection.
    verification_rows = []
    for record in records:
        verification_rows.append({
            key: value for key, value in {
                **record,
                "month": record["date_utc"][:7],
                "selected_image": "",
                "candidate_image": str(record["candidate_path"]),
            }.items() if key in MANIFEST_FIELDS
        })

    verification_dir = args.output / "verification"
    verification_dir.mkdir(parents=True, exist_ok=True)
    verification_path = verification_dir / f"{brand}_verification.csv"
    with verification_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(verification_rows)

    verified_records = [r for r in records if r["verification_status"] == "verified"]
    selected = select_diverse_images(verified_records, args.images_per_brand)
    if len(selected) != args.images_per_brand:
        raise RuntimeError(
            f"{brand}: found only {len(selected)} verified images; "
            f"{args.images_per_brand} required. Increase --scan-posts or widen the date range."
        )

    selected_dir = args.output / "selected" / brand
    selected_dir.mkdir(parents=True, exist_ok=True)
    for path in image_files(selected_dir):
        path.unlink()
    manifest_rows = []

    for index, record in enumerate(selected, start=1):
        date_part = record["date_utc"][:10]
        extension = record["candidate_path"].suffix.lower()
        filename = f"{index:03d}_{date_part}_{record['post_shortcode']}{extension}"
        destination = selected_dir / filename
        shutil.copy2(record["candidate_path"], destination)
        manifest_rows.append({
            key: value for key, value in {
                **record,
                "month": record["date_utc"][:7],
                "selected_image": str(destination),
                "candidate_image": str(record["candidate_path"]),
            }.items() if key in MANIFEST_FIELDS
        })

    manifest_path = args.output / "selected" / f"{brand}_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    return {
        "brand": brand,
        "account_url": account_url,
        "eligible_posts_scanned": len(eligible_posts),
        "candidate_posts_downloaded": len(downloaded_posts),
        "readable_candidate_images": len(records),
        "verified_images": len(verified_records),
        "review_images": sum(r["verification_status"] == "review" for r in records),
        "rejected_images": sum(r["verification_status"] == "rejected" for r in records),
        "selected_images": len(selected),
        "months_covered": sorted({row["month"] for row in manifest_rows}),
        "verification_manifest": str(verification_path),
        "manifest": str(manifest_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("instagram_brand_images"))
    parser.add_argument("--brands", nargs="+", choices=list(BRAND_PROFILES),
                        default=list(BRAND_PROFILES))
    parser.add_argument("--images-per-brand", type=int, default=15)
    parser.add_argument("--scan-posts", type=int, default=600)
    parser.add_argument("--candidate-posts", type=int, default=30)
    parser.add_argument("--delay", type=float, default=5.0)
    parser.add_argument("--since", help="Inclusive UTC date (default: two years ago)")
    parser.add_argument("--until", help="Inclusive UTC date, e.g. 2026-11-07")
    parser.add_argument(
        "--vlm-provider",
        choices=["auto", "ollama", "openai"],
        default="auto",
        help="API dialect; auto detects Ollama when the endpoint uses port 11434",
    )
    parser.add_argument(
        "--vlm-base-url",
        default=os.getenv("VLM_BASE_URL", "http://localhost:8000/v1"),
        help="OpenAI-compatible Qwen endpoint (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--vlm-model",
        default=os.getenv("VLM_MODEL", "Qwen/Qwen3.5-9B"),
        help="Vision model name exposed by the endpoint",
    )
    parser.add_argument(
        "--vlm-api-key",
        default=os.getenv("VLM_API_KEY", ""),
        help="API key if required; otherwise leave empty for a local server",
    )
    parser.add_argument("--vlm-timeout", type=int, default=120)
    args = parser.parse_args()

    if min(args.images_per_brand, args.scan_posts, args.candidate_posts) < 1:
        parser.error("Image and post limits must be positive")
    if args.delay < 0:
        parser.error("--delay cannot be negative")
    if args.vlm_timeout < 1:
        parser.error("--vlm-timeout must be positive")

    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    loader = make_loader(args.output, args.delay)
    report = {"completed": [], "errors": []}

    for brand in args.brands:
        try:
            report["completed"].append(
                collect_brand(brand, BRAND_PROFILES[brand], args, loader)
            )
        except (instaloader.exceptions.LoginRequiredException,
                instaloader.exceptions.ConnectionException) as exc:
            report["errors"].append({"brand": brand, "error": str(exc)})
            print(f"Stopping collection: Instagram access failed for {brand}: {exc}")
            break  # Do not repeatedly retry after a platform/network block.
        except Exception as exc:
            report["errors"].append({"brand": brand, "error": str(exc)})
            print(f"{brand}: {exc}")

    report_path = args.output / "collection_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Collection report: {report_path}")


if __name__ == "__main__":
    main()