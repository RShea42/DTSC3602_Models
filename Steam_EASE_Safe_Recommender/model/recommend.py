"""Command line: recommendations for a Steam profile.

  python model/recommend.py <steam id | profile link | custom profile name> [--n 10] [--max-price 30]

Only public game details are read; no password or sign-in is ever needed.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recommender import SteamError, for_profile  # noqa: E402


def show(title, cards):
    print(f"\n{title}\n" + "-" * len(title))
    for i, c in enumerate(cards, 1):
        price = (c["price_text"] or "price n/a") + (f" ({c['discount']}% off)" if c.get("discount") else "")
        rating = f"{c['rating']}% positive" if c["rating"] is not None else ""
        print(f"{i:2d}. {c['title']}  [{price}{', ' + rating if rating else ''}]")
        why = (f"new for you: {c['new_tag']}; " if c.get("new_tag") else "") + \
              ("because you enjoyed " + " and ".join(c["because"]) if c["because"] else "")
        print(f"    {why}\n    {c['store_url']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-price", type=float, default=None)
    a = ap.parse_args()
    try:
        r = for_profile(a.profile, n=a.n, max_price=a.max_price)
    except SteamError as e:
        sys.exit(str(e))
    print(f"{r['player']['name']}: {r['n_owned']} games owned, {r['n_used']} used for taste. "
          f"Top tags: {', '.join(r['top_tags'])}. Favourites: {', '.join(r['favorites'][:3])}")
    show("For you", r["for_you"])
    show("Try something new", r["try_new"])
