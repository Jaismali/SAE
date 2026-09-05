"""Layer selection for SAE-based concept extraction.

Computes the theoretical target layer from a model's total depth and a
locked relative-depth band, then reconciles that theoretical target
against the layers actually available in a real SAE release. Any
mismatch between "theoretical nearest layer" and "actual layer used"
is treated as a deviation that must be logged with checkable evidence,
never silently substituted.
"""

from dataclasses import dataclass, field


# --- Locked project constants (Gemma 3 1B-IT, Gemma Scope 2 release) ---

MODEL_TOTAL_LAYERS = 26
RELATIVE_DEPTH_BAND = (0.50, 0.60)

# Confirmed via sae_lens.loading.pretrained_saes_directory
# .get_pretrained_saes_directory() for release "gemma-scope-2-1b-it-res"
# (repo: google/gemma-scope-2-1b-it). 52 keys total = 4 layers x 4 widths
# x 3 L0 levels + 4 seed variants -- confirms a complete, non-truncated
# manifest was captured, not a partial listing.
AVAILABLE_LAYERS_IN_RELEASE = [7, 13, 17, 22]

THEORETICAL_NEAREST_TARGET_LAYER = 14
ACTUAL_TARGET_LAYER = 13

# THEORETICAL_IN_BAND_LAYERS corrected to {13,14,15} -- the original
# Month 2 session incorrectly included layer 16 (61.5% depth) by
# conflating the precise 50-60% band with Pack 1's looser "13-16"
# illustrative phrasing. See conversation log for the arithmetic error.
THEORETICAL_IN_BAND_LAYERS = [13, 14, 15]

# Layer 16 was queried against AVAILABLE_LAYERS_IN_RELEASE during the
# original (mistaken) {13,14,15,16} framing and genuinely came back
# absent. That fact is true and worth keeping as a footnote, but it is
# NOT evidence for the in-band deviation claim, since 16 (61.5% depth)
# was never a legitimate in-band candidate.
LAYER_16_HISTORICAL_NOTE = (
    "Layer 16 (61.5% relative depth) was queried against the real "
    "release during an earlier, incorrect {13,14,15,16} framing and "
    "confirmed absent. This is a true fact about the release but is "
    "excluded from the in-band deviation claim, since layer 16 was "
    "never actually within the locked 50-60% band."
)


def compute_nearest_layer_in_band(
    total_layers: int,
    band: tuple[float, float] = RELATIVE_DEPTH_BAND,
    rounding: str = "nearest",
) -> int:
    """Compute the theoretical target layer from relative depth alone.

    This does not know about any real SAE release's availability -- it
    answers "what layer would we pick if every layer had an SAE."

    Args:
        total_layers: total number of layers in the model.
        band: (low, high) fraction of total depth defining the target band.
        rounding: "nearest" (default) rounds the band midpoint to the
            nearest integer layer. No other rounding rule is implemented
            yet; passing anything else raises ValueError rather than
            silently falling back to "nearest".

    Returns:
        The theoretical target layer index (0-indexed, matching the
        convention used by AVAILABLE_LAYERS_IN_RELEASE).
    """
    if rounding != "nearest":
        raise ValueError(
            f"Unsupported rounding rule: {rounding!r}. "
            "Only 'nearest' is implemented; add new rules explicitly "
            "rather than defaulting to one silently."
        )

    low_fraction, high_fraction = band
    band_midpoint_fraction = (low_fraction + high_fraction) / 2
    midpoint_layer = band_midpoint_fraction * total_layers
    return round(midpoint_layer)


def find_available_layers_in_band(
    available_layers: list[int],
    total_layers: int,
    band: tuple[float, float] = RELATIVE_DEPTH_BAND,
) -> list[int]:
    """Return every available layer whose relative depth falls in-band.

    Used to prove absence, not just report it: a deviation record should
    show every in-band candidate that was checked and rejected, not only
    the single theoretical nearest one.
    """
    low_fraction, high_fraction = band
    return [
        layer
        for layer in available_layers
        if low_fraction <= (layer / total_layers) <= high_fraction
    ]


def relative_depth(layer: int, total_layers: int) -> float:
    """Relative depth of a layer as a fraction of total model depth."""
    return layer / total_layers


@dataclass
class LayerDeviationRecord:
    """Auditable record of a theoretical-vs-actual layer substitution."""

    theoretical_target_layer: int
    actual_target_layer: int
    available_layers_in_release: list[int]
    in_band_candidates_checked: list[int] = field(default_factory=list)
    in_band_candidates_confirmed_absent: list[int] = field(default_factory=list)
    layer_16_checked_but_out_of_band_note: str = ""
    total_layers: int = MODEL_TOTAL_LAYERS
    band: tuple[float, float] = RELATIVE_DEPTH_BAND

    def as_log_string(self) -> str:
        theoretical_depth = relative_depth(
            self.theoretical_target_layer, self.total_layers
        )
        actual_depth = relative_depth(self.actual_target_layer, self.total_layers)
        low, high = self.band

        lines = [
            "LAYER SELECTION DEVIATION",
            f"  Locked relative-depth band: {low:.0%}-{high:.0%} "
            f"of {self.total_layers} total layers",
            f"  Theoretical nearest target layer: {self.theoretical_target_layer} "
            f"({theoretical_depth:.1%} relative depth)",
            f"  Layers actually available in release: "
            f"{self.available_layers_in_release}",
            f"  In-band candidates checked: {self.in_band_candidates_checked}",
            f"  In-band candidates confirmed absent: "
            f"{self.in_band_candidates_confirmed_absent}",
            f"  Actual target layer used: {self.actual_target_layer} "
            f"({actual_depth:.1%} relative depth)",
        ]

        if self.layer_16_checked_but_out_of_band_note:
            lines.append(
                f"  Footnote (not part of the in-band claim): "
                f"{self.layer_16_checked_but_out_of_band_note}"
            )

        if actual_depth < low or actual_depth > high:
            lines.append(
                f"  WARNING: actual target layer falls OUTSIDE the locked band."
            )
        else:
            lines.append(
                f"  Actual target layer remains within the locked band "
                f"(at the {'lower' if actual_depth <= (low + high) / 2 else 'upper'} "
                f"edge)."
            )

        return "\n".join(lines)


def build_deviation_record(
    theoretical_target_layer: int = THEORETICAL_NEAREST_TARGET_LAYER,
    actual_target_layer: int = ACTUAL_TARGET_LAYER,
    available_layers_in_release: list[int] = None,
    total_layers: int = MODEL_TOTAL_LAYERS,
    band: tuple[float, float] = RELATIVE_DEPTH_BAND,
) -> LayerDeviationRecord:
    """Build the full deviation record for the layer-14-to-13 substitution.

    Checks every genuinely in-band candidate explicitly (not just the
    theoretical target) so the record proves that layers 14 and 15 were
    each confirmed absent from the release, rather than only layer 14.

    Layer 16 is deliberately excluded from the in-band claim: it sits
    at 61.5% relative depth, outside the locked 50-60% band, despite an
    earlier session error that treated it as in-band. See
    LAYER_16_HISTORICAL_NOTE for the (still-true, but out-of-band) fact
    that it was also queried and found absent.
    """
    if available_layers_in_release is None:
        available_layers_in_release = AVAILABLE_LAYERS_IN_RELEASE

    in_band_candidates_checked = [
        layer
        for layer in range(total_layers)
        if band[0] <= relative_depth(layer, total_layers) <= band[1]
    ]
    in_band_candidates_confirmed_absent = [
        layer
        for layer in in_band_candidates_checked
        if layer not in available_layers_in_release
    ]

    layer_16_note = ""
    if 16 not in in_band_candidates_checked and 16 not in available_layers_in_release:
        layer_16_note = LAYER_16_HISTORICAL_NOTE

    return LayerDeviationRecord(
        theoretical_target_layer=theoretical_target_layer,
        actual_target_layer=actual_target_layer,
        available_layers_in_release=available_layers_in_release,
        in_band_candidates_checked=in_band_candidates_checked,
        in_band_candidates_confirmed_absent=in_band_candidates_confirmed_absent,
        layer_16_checked_but_out_of_band_note=layer_16_note,
        total_layers=total_layers,
        band=band,
    )


if __name__ == "__main__":
    record = build_deviation_record()
    print(record.as_log_string())
