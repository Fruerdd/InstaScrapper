import argparse
import csv
import re
import time
from itertools import islice
from pathlib import Path
from urllib.parse import urlparse

import instaloader


def username_from_url(account_url: str) -> str:
    """Extract and validate a username from an Instagram profile URL."""
    parsed = urlparse(account_url)
    hostname = (parsed.hostname or "").lower()

    if hostname not in {"instagram.com", "www.instagram.com"}:
        raise ValueError("Please provide a full instagram.com profile URL.")

    parts = [part for part in parsed.path.split("/") if part]

    if len(parts) != 1:
        raise ValueError(
            "Use a profile URL such as "
            "https://www.instagram.com/brandname/, not a post URL."
        )

    username = parts[0]

    if username.lower() in {"p", "reel", "reels", "explore", "stories"}:
        raise ValueError("The supplied URL is not an account profile URL.")

    if not re.fullmatch(r"[A-Za-z0-9._]+", username):
        raise ValueError("The Instagram username contains invalid characters.")

    return username


def main():
    parser = argparse.ArgumentParser(
        description="Download images from posts on a public Instagram profile."
    )
    parser.add_argument("account_url", help="Public Instagram profile URL")
    parser.add_argument(
        "--output",
        default="instagram_images",
        help="Output directory (default: instagram_images)",
    )
    parser.add_argument(
        "--max-posts",
        type=int,
        default=50,
        help="Maximum number of recent posts to inspect (default: 50)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
        help="Delay between processed posts in seconds (default: 5)",
    )
    args = parser.parse_args()

    if args.max_posts < 1:
        parser.error("--max-posts must be at least 1")
    if args.delay < 0:
        parser.error("--delay cannot be negative")

    username = username_from_url(args.account_url)
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = instaloader.Instaloader(
        dirname_pattern=str(output_dir / "{target}"),
        filename_pattern="{date_utc:%Y%m%dT%H%M%SZ}_{shortcode}",
        download_pictures=True,
        download_videos=False,
        download_video_thumbnails=False,
        download_comments=False,
        download_geotags=False,
        save_metadata=False,
        post_metadata_txt_pattern="",  # Do not save captions as text files
        quiet=False,
    )

    manifest_rows = []

    try:
        profile = instaloader.Profile.from_username(loader.context, username)

        if profile.is_private:
            raise SystemExit(
                "This is a private profile. The script only processes "
                "publicly accessible profiles."
            )

        # Limit the iterator before iterating; no login or session is loaded.
        posts = islice(profile.get_posts(), args.max_posts)

        for post in posts:
            # Skip video-only posts. Images in mixed carousel posts can still
            # be downloaded because video downloading is disabled above.
            if post.is_video:
                print(f"Skipping video-only post: {post.shortcode}")
                continue

            print(f"Downloading images from post: {post.shortcode}")
            loader.download_post(post, target=username)

            manifest_rows.append(
                {
                    "account": username,
                    "shortcode": post.shortcode,
                    "date_utc": post.date_utc.isoformat(),
                    "post_url": f"https://www.instagram.com/p/{post.shortcode}/",
                    "post_type": post.typename,
                }
            )

            time.sleep(args.delay)

    except instaloader.exceptions.ProfileNotExistsException as exc:
        raise SystemExit(f"Instagram profile not found: {username}") from exc
    except instaloader.exceptions.LoginRequiredException as exc:
        raise SystemExit(
            "Instagram requires login for this request. This script does not "
            " bypass login restrictions; use an officially permitted data "
            "source instead."
        ) from exc
    except instaloader.exceptions.ConnectionException as exc:
        raise SystemExit(
            "Instagram refused or interrupted the connection, possibly due "
            "to rate limiting. Stop the collection rather than repeatedly "
            f"retrying. Details: {exc}"
        ) from exc

    manifest_path = output_dir / f"{username}_posts.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "account",
                "shortcode",
                "date_utc",
                "post_url",
                "post_type",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nSaved {len(manifest_rows)} image-containing posts.")
    print(f"Images: {output_dir / username}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()