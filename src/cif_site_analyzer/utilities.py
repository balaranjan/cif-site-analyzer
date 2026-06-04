import os
import re
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from cif_reader import read_cif
from cif_reader.base import _parse_formula
from collections import defaultdict


def find_sites_with_same_wyckoff_symbols(cifpath):
    shared = {}
    sites = {}

    cif = read_cif(cifpath)
    for site in cif.site_data:
        wyckoff = f"{int(site['multiplicity'])}{site['Wyckoff_symbol']}"
        if wyckoff not in sites:
            sites[wyckoff] = [site["coordinates"]]
        else:
            current_coords = sites[wyckoff]
            duplicate = False
            for coord in current_coords:
                if np.allclose(coord, site["coordinates"]):
                    duplicate = True

            if not duplicate:
                current_coords.append(site["coordinates"])
                sites[wyckoff] = current_coords

    for k, v in sites.items():
        if len(v) > 1:
            shared[k] = np.array(v)

    return shared


def get_site_label(site, shared):
    wyckoff = f"{int(site['multiplicity'])}{site['Wyckoff_symbol']}"

    if wyckoff in shared:
        if len(shared):
            order = shared[wyckoff]
            ind = [
                [
                    i + 1,
                    np.linalg.norm(np.array(site["coordinates"]) - order[i]),
                ]
                for i in range(order.shape[0])
            ]
            ind = sorted(ind, key=lambda x: x[1])
            wyckoff = f"{wyckoff}{ind[0][0]}"
    #     return wyckoff
    # else:
    return wyckoff


def get_data_row(cifpath, shared_wyckoffs=None):

    cif = read_cif(cifpath)
    site_formula = []
    for site in cif.site_data:
        comp = site["symbol"]
        wyckoff = get_site_label(site, shared_wyckoffs)
        if site["occupancy"] != 1.0:
            comp += str(site["occupancy"])
        site_formula.append([comp, wyckoff, site["coordinates"]])

    # same site
    site_formula_r = {}
    for i in range(len(site_formula)):
        icoordinates = site_formula[i][2]
        iformula = site_formula[i][0]
        iwyckoff = site_formula[i][1]

        if iwyckoff not in site_formula_r:
            site_formula_r[iwyckoff] = [iformula, icoordinates]
        else:
            jformula, jcoordinates = site_formula_r[iwyckoff]
            if np.allclose(icoordinates, jcoordinates):
                site_formula_r[iwyckoff] = [jformula + iformula, icoordinates]

    data = {
        # 'PCD Entry No.': cif['#_database_code_PCD'],
        "Filename": cifpath.split(os.sep)[-1],
        "Formula": cif.formula,
        "Notes": "",
        "Num Elements": len(cif.formula_dict),
    }
    for k, v in site_formula_r.items():
        data[k] = v
    return data


def get_site_element_dist(
    dataframe: pd.DataFrame, site: str = None
) -> pd.DataFrame:
    element_stats = defaultdict(float)

    for _, row in dataframe.iterrows():
        if row["Formula"] == "":
            continue

        if site is None:
            formula = _parse_formula(str(row["Formula"]))
        else:
            formula = _parse_formula(str(row[site]))

        for k, v in formula.items():
            element_stats[k] += v

    return element_stats


def format_wyckoff_site_label(site_label, italicize=False):
    ws = [v for v in re.split(r"\d+", site_label) if v != ""][0]
    nums = [s for s in site_label.split(ws) if s != ""]
    if italicize:
        site_label = nums[0] + f"${ws}$"
    else:
        site_label = nums[0] + f"{ws}"
    if len(nums) > 1:
        site_label += f" ({''.join(nums[1:])})"
    return site_label


def get_ws(s):
    s = re.split(r"\d", s)
    return [_s for _s in s if _s != ""][0]


def format_formula(formula):
    formula = _parse_formula(formula)

    new_formula = ""
    for k, v in formula.items():
        if v == 1.0:
            new_formula += k
        else:
            if abs(int(v) - v) < 1e-3:
                new_formula += k + "$_{%s}$" % int(v)
            else:
                new_formula += k + "$_{%s}$" % v

    return new_formula


def format_mixed_formula(formula):
    """Converts a string like 'Cr0.16Mo0.38Co0.46' to 'Cr 0.16, Mo 0.38, Co
    0.46'."""
    parts = re.findall(r"([A-Z][a-z]*)([0-9.]+)", formula)
    return ", ".join([f"{el} {amt}" for el, amt in parts])


if __name__ == "__main__":
    form = "Cr0.16Mo0.38Co0.46"

    print(format_mixed_formula(form))
    print(format_formula(form))


def merge_images(parent_dir):

    print("Merging images")

    png_files = glob.glob(os.path.join(parent_dir, "G*.png"))

    if not png_files:
        return

    def sort_key(path):
        fname = os.path.basename(path)
        return int(fname[1]) if len(fname) > 1 and fname[1].isdigit() else 0

    png_files_sorted = sorted(png_files, key=sort_key)
    images = [mpimg.imread(path) for path in png_files_sorted]

    # vertically stacked
    fig_col, axes_col = plt.subplots(
        len(images),
        1,
        figsize=(5 * len(images), 5),
        gridspec_kw={"hspace": 0.1, "wspace": 0},
    )
    if len(images) == 1:
        axes_col = [axes_col]
    for ax, img in zip(axes_col, images):
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(parent_dir, "col.png"), bbox_inches="tight", dpi=300
    )
    plt.close(fig_col)

    # horizontally stacked
    fig_row, axes_row = plt.subplots(
        1,
        len(images),
        figsize=(5 * len(images), 5),
        gridspec_kw={"hspace": 0, "wspace": 0},
    )
    if len(images) == 1:
        axes_row = [axes_row]
    for ax, img in zip(axes_row, images):
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(parent_dir, "row.png"), bbox_inches="tight", dpi=300
    )
    plt.close(fig_row)
