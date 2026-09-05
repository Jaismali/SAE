"""
Lists every release name in SAELens's pretrained-SAE directory that
mentions 'gemma-3' or 'gemma-scope-2', so we can find the correct exact
release string for our locked target (gemma-3-1b-it) instead of
guessing at naming conventions from documentation snippets.
"""

from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

if __name__ == "__main__":
    directory = get_pretrained_saes_directory()

    print(f"Total releases known to this SAELens installation: {len(directory)}\n")

    matches = {
        release_name: info
        for release_name, info in directory.items()
        if "gemma-3" in release_name.lower() or "gemma-scope-2" in release_name.lower()
    }

    if not matches:
        print("No releases found matching 'gemma-3' or 'gemma-scope-2'.")
        print("This SAELens version may not yet support Gemma Scope 2.")
    else:
        print(f"Found {len(matches)} matching release(s):\n")
        for release_name, info in matches.items():
            print(f"  release: {release_name!r}")
            # Print the raw underlying data directly, rather than guessing
            # attribute names -- avoids a repeat of the earlier guess-and-fail
            # cycle if this object's shape doesn't match assumptions.
            info_dict = info.__dict__ if hasattr(info, "__dict__") else info
            for key, value in info_dict.items():
                if key == "saes_map" and isinstance(value, dict):
                    sae_ids = list(value.keys())
                    print(f"    {key}: {len(sae_ids)} entries, first few: {sae_ids[:5]}")
                else:
                    print(f"    {key}: {value}")
            print()
