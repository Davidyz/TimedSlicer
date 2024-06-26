import argparse
import os
import re
from typing import List, Sequence

import cv2
import numpy as np
import tqdm


def parse_sequence_ordering(images: Sequence, pattern: str) -> Sequence[str]:
    result: List[str] = []
    pattern_vals = [int(i.strip()) for i in pattern.split(",")]
    for seq in pattern_vals:
        step_size = len(images) // (abs(seq) - 1)
        if seq < 0:
            step_size = -step_size
        result.extend(images[::step_size])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "images", nargs="+", help="Paths to images in the desired order."
    )
    parser.add_argument(
        "--horizontal",
        action="store_true",
        default=False,
        help="Use horizontal slices.",
    )
    parser.add_argument(
        "--reversed",
        action="store_true",
        default=False,
        help="Generate slices in reversed order.",
    )
    parser.add_argument(
        "--output", type=str, default="out.jpg", help="Output image path."
    )
    parser.add_argument(
        "--interval_size",
        type=int,
        default=-1,
        help="When supplying a large number of images, use this option to specify the interval between two chosen images.",
    )
    parser.add_argument(
        "--slice_count",
        type=int,
        default=-1,
        help="Number of sliced used for the final image.",
    )
    parser.add_argument(
        "--slice_pattern",
        type=str,
        default="",
        help="RegEx: -?%d+(,-?%d+)*, a sequence of slice patterns that allows non-linear slice ordering.",
    )
    args = parser.parse_args()
    args.images.sort()
    args.images = [i for i in args.images if os.path.isfile(i)]
    assert (
        (args.interval_size > 0) ^ (args.slice_count > 0) ^ (args.slice_pattern != "")
    ), f"{args.interval_size=}, {args.slice_count=}, {args.slice_pattern=}"

    if args.interval_size > 0:
        args.images = args.images[:: args.interval_size]
    elif args.slice_count > 0:
        interval_size = len(args.images) // (args.slice_count - 1)
        images = args.images[0:-1:interval_size]
        if len(images) < args.slice_count:
            images.append(args.images[-1])
        args.images = images
    elif args.slice_pattern != "":
        if re.match(r"-?\d+(,-?\d+)*", args.slice_pattern) is None:
            raise ValueError(f"Invalid pattern: {args.slice_pattern}")
        args.images = parse_sequence_ordering(args.images, args.slice_pattern)

    is_horizontal: bool = args.horizontal
    use_reversed: bool = args.reversed

    canvas = np.zeros_like(cv2.imread(args.images[0]), dtype=np.uint8)
    step_width = canvas.shape[1] // len(args.images)
    if is_horizontal:
        step_width = canvas.shape[0] // len(args.images)

    slice_range = (
        range(len(args.images))
        if not use_reversed
        else range(len(args.images) - 1, -1, -1)
    )
    for i_img, i_slice in tqdm.tqdm(enumerate(slice_range), total=len(args.images)):
        image = cv2.imread(args.images[i_img])
        if image is None:
            print(args.images[i_img])
        if is_horizontal:
            canvas[step_width * i_slice : step_width * (i_slice + 1), :] = image[
                step_width * i_slice : step_width * (i_slice + 1), :
            ]
        else:
            canvas[:, step_width * i_slice : step_width * (i_slice + 1)] = image[
                :, step_width * i_slice : step_width * (i_slice + 1)
            ]

    while np.count_nonzero(canvas[:, -1]) == 0:
        canvas = canvas[:, :-1]

    while np.count_nonzero(canvas[-1, :]) == 0:
        canvas = canvas[:-1, :]
    print(f"Writing output image to {args.output}")
    cv2.imwrite(args.output, canvas)
