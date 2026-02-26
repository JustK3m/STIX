import numpy as np
import numpy.typing
import get_file_info as get
import matplotlib.pyplot as plt
from astropy.modeling.models import custom_model

"""
Perform a flux- (i.e. area-) conserving rebinning of a histogram,
given its edges and a set of new edges.
The __name__ ... section has an example using a power law.
Credits to W Setterberg, Shane Maloney, Kris Cooper & Dan Ryan, Jan 2023
"""


def flux_conserving_rebin(
        old_edges: np.typing.ArrayLike,
        old_values: np.typing.ArrayLike,
        new_edges: np.typing.ArrayLike,
) -> np.ndarray:
    """
    Rebin a histogram by performing a flux-conserving rebinning.
    The total area of the histogram is conserved.
    Adjacent bins are proportionally interpolated for new edges that do not line up.
    Don't make the new edges too finely spaced;
    don't make a new bin fall inside of an old one completely.
    """
    old_edges = np.array(np.sort(old_edges))
    new_edges = np.array(np.sort(new_edges))
    nd = np.diff(new_edges)
    od = np.diff(old_edges)

    if (new_edges[0] < old_edges[0]) or (new_edges[-1] > old_edges[-1]):
        raise ValueError('New edges cannot fall outside range of old edges.')

    # if np.all(nd == od):
    #     return old_values[:]
    
    if len(nd) == len(od) and np.allclose(nd, od, rtol=1e-5):
        return old_values[:len(nd)]


    orig_flux = od * old_values
    ret = np.zeros(new_edges.size - 1)
    for i in range(ret.size):
        ret[i] = interpolate_new_bin(
            original_area=orig_flux,
            old_edges=old_edges,
            new_left=new_edges[i],
            new_right=new_edges[i + 1]
        )

    return ret


def proportional_interp_single_bin(
        left_edge: float,
        right_edge: float,
        interp: float
) -> tuple[float, float]:
    """
    say what portion of a histogram bin belongs on the left and right
    of an edge to interpolate.
    """
    denom = right_edge - left_edge
    right_portion = (right_edge - interp) / denom
    left_portion = (interp - left_edge) / denom
    return left_portion, right_portion


def bounding_interpolate_indices(
        old_edges: np.ndarray,
        left: float,
        right: float
) -> tuple[int, int]:
    """
    find the indices of the old edges that bound the new left
    and right edges.
    """
    indices = np.arange(old_edges.size)
    new_left = indices[old_edges <= left][-1]
    new_right = indices[old_edges >= right][0]
    return new_left, new_right


def interpolate_new_bin(
        original_area: np.array,
        old_edges: np.array,
        new_left: float,
        new_right: float
) -> float:
    """
    interpolate the new bin value given old edges, new edges,
    and the old flux (aka area).
    """
    oa = original_area
    oe = old_edges

    old_start_idx, old_end_idx = bounding_interpolate_indices(
        oe, new_left, new_right
    )

    # portion of edge bins that get grouped with the new bin
    _, left_partial_prop = proportional_interp_single_bin(
        left_edge=oe[old_start_idx],
        right_edge=oe[old_start_idx + 1],
        interp=new_left
    )
    left_partial_area = left_partial_prop * oa[old_start_idx]

    right_partial_prop, _ = proportional_interp_single_bin(
        left_edge=oe[old_end_idx - 1],
        right_edge=oe[old_end_idx],
        interp=new_right
    )
    right_partial_area = right_partial_prop * oa[old_end_idx - 1]

    partial_slice = slice(old_start_idx + 1, old_end_idx - 1)
    have_bad_slice = (partial_slice.start > partial_slice.stop)
    if have_bad_slice:
        raise ValueError('Your new bins are too fine. Use coarser bins.')

    between_area = oa[partial_slice].sum()

    delta = new_right - new_left
    new_bin_value = (left_partial_area + between_area + right_partial_area) / delta
    return new_bin_value


@custom_model
def fitting_photons(x, method=0., p1=1., p2=1., p3=1, p4=1., p5=1., index_start=0.):
    """
    :param x: energy bands data from spectrogram
    :param method: corresponding number of method used:
                0: Power Law
                1: Broken Power Law
                2: Gaussian
                3: Polynomial
                4: Exponential
                5: Single Power Law x Exponential
                6: Logistic regression
                7: Lorentz
                8: Moffat
                9: Voigt Profile
    :param p1: first parameter
    :param p2: second parameter
    :param p3: third parameter
    :param p4: fourth parameter
    :param p5: fifth parameter
    :param index_start: index of the position of the first energy band
    :return rebinned_counts:
    """

    srm_file = './stx_srm_2021feb14_0140_0155.fits'
    fit_method = "Power Law"

    photon_bins, channel_bins, srm, energy_bands = get.get_srm(srm_file)

    low_bd = int(index_start) if index_start > 0 else 0
    num_old, num_new = np.size(x), energy_bands - low_bd
    print("nums: ", num_old, num_new)
    up_bd = low_bd + num_old
    print("bands: ", low_bd, up_bd)

    print(srm[0, 5:21])
    srm = srm[:, low_bd:up_bd]
    print(srm[0, :])

    #if num_old < num_new:
    #    num_new = int(num_old)

    # FIXME: Check if both limits O - 150 are correct to use
    old_edges = np.linspace(0, 150, num=num_old + 1)  # num=32 + 1
    new_edges = np.linspace(0, 150, num=num_new + 1)  # num=20 + 1

    if method == 0:    # 0. Power Law
        #old_values = ((x / p1) ** (-p2)) 
        safe_x = np.where(x == 0, 1e-6, x)
        old_values = ((safe_x / p1) ** (-p2))
  # FIXME: If p1 is not fixed, the fitting tries to divide by 0 in Power Law
    elif method == 1:  # 1. Broken Power Law
        old_values = x
        print("Not done yet.")
    elif method == 2:  # 2. Gaussian
        old_values = x
        print("Not done yet.")
    elif method == 3:  # 3. Polynomial
        old_values = x
        print("Not done yet.")
    elif method == 4:  # 4. Exponential
        old_values = np.exp(p1 - x / p2)
    elif method == 5:  # 5. Single Power Law x Exponential
        old_values = (p5 * (x / p2) ** p1) * (np.exp(p3 - x / p4))
    elif method == 6:  # 6. Logistic regression
        old_values = 1 / (1 + np.exp(-x))
    elif method == 7:  # 7. Lorentz
        old_values = x
        print("Not done yet.")
    elif method == 8:  # 8. Moffat
        old_values = x
        print("Not done yet.")
    elif method == 9:  # 9. Voigt Profile
        old_values = x
        print("Not done yet.")
    else:              # Else: if no method is recognized, keeps by default the input data.
        old_values = x[:]
        print("No method recognized. Try with another argument.")

    # plt.figure(2)
    # plt.plot(x, old_values)

    print(np.size(x), energy_bands, low_bd, up_bd, np.size(srm, 0), np.size(srm, 1),
          num_old, num_new, np.size(old_edges), np.size(new_edges))

    new_values = flux_conserving_rebin(
        old_edges=old_edges,
        old_values=old_values,
        new_edges=new_edges
    )
    new_counts = np.matrix(new_values) * srm.T
    print("ICI : ", np.size(old_values), np.size(new_counts))

    rebinned_counts = np.zeros(num_old)
    for eb in range(num_old):  # i in range(i.e. 20 or 32)
        rebinned_counts[eb] = float(new_counts[0, eb])

    # plt.plot(x, rebinned_counts)
    # plt.show()

    if __name__ == '__main__':
        original_flux = compute_binned_flux(old_edges, old_values)
        new_flux = compute_binned_flux(new_edges, new_values)
        mids = old_edges[:-1] + np.diff(old_edges) / 2

        print(f'original flux:     {original_flux:.4f}')
        print(f'rebinned flux:     {new_flux:.4f}')
        print(f'amount off:        {np.abs(original_flux - new_flux):.2e}')
        print(f'machine precision: {np.finfo(float).eps:.2e}')
        print(';)')

        fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
        ax.stairs(mids, old_edges, label='original')
        ax.stairs(new_values, new_edges, label='rebinned')
        ax.stairs(rebinned_counts, new_edges, label='rebinned with srm')
        ax.set(
            xscale='log',
            yscale='log',
            xlabel='$x$',
            ylabel=str(fit_method),
            title='Rebin linear-spaced bins to log-spaced bins'
        )
        ax.set_xlim(4, 150)
        ax.set_ylim(10, 10 ** 6)
        ax.legend()
        plt.show()

    return rebinned_counts


if __name__ == '__main__':

    def compute_binned_flux(edges, values):
        return np.sum(values * np.diff(edges))

    stix_srm_file = './stx_srm_2021feb14_0140_0155.fits'
    fitting_method = "Power Law"

    rebinned_counts_model_test = fitting_photons(p1=1., p2=1.)

    print(rebinned_counts_model_test)
