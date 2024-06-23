# Ayal_Kaabia, 322784760
# Sami_Serhan, 327876298

import cv2
import numpy as np
import os
import shutil
import sys


def get_transform(matches, is_affine):
    src_points, dst_points = matches[:, 0], matches[:, 1]

    # convert point to type float32
    f_src_points = src_points.astype(np.float32)
    f_dst_points = dst_points.astype(np.float32)

    if is_affine:   # get the Affine Transform matrix
        T = cv2.getAffineTransform(f_src_points, f_dst_points)
    else:   # get the Perspective Transform matrix
        T = cv2.getPerspectiveTransform(f_src_points, f_dst_points)

    return T


def stitch(img1, img2):
   # create a mask from the second image where the pixels are not zero
   mask = np.any(img2 != 0, axis=-1)

   if len(mask.shape) == 2:
       mask = np.stack([mask] * 3, axis=-1)

   stitched_img = np.where(mask, img2, img1)
   return stitched_img


def inverse_transform_target_image(target_img, original_transform, output_size):
    if original_transform.shape == (2, 3):  # affine transformation
        result = cv2.warpAffine(target_img, original_transform, output_size, flags=cv2.WARP_INVERSE_MAP)

    else:  # Homography
        result = cv2.warpPerspective(target_img, original_transform, output_size, flags=cv2.WARP_INVERSE_MAP)
    return result


def prepare_puzzle(puzzle_dir):
    edited = os.path.join(puzzle_dir, 'abs_pieces')
    if os.path.exists(edited):
        shutil.rmtree(edited)
    os.mkdir(edited)

    affine = 4 - int("affine" in puzzle_dir)

    matches_data = os.path.join(puzzle_dir, 'matches.txt')
    n_images = len(os.listdir(os.path.join(puzzle_dir, 'pieces')))

    matches = np.loadtxt(matches_data, dtype=np.int64).reshape(n_images - 1, affine, 2, 2)

    return matches, affine == 3, n_images


if __name__ == '__main__':
    lst = ['puzzle_affine_1', 'puzzle_affine_2', 'puzzle_homography_1']

    for puzzle_dir in lst:
        # extract the file path
        print(f'Starting {puzzle_dir}')

        puzzle = os.path.join('puzzles', puzzle_dir)

        pieces_pth = os.path.join(puzzle, 'pieces')
        edited = os.path.join(puzzle, 'abs_pieces')

        matches, is_affine, n_images = prepare_puzzle(puzzle)

        final_puzzle = cv2.imread(os.path.join(pieces_pth, 'piece_1.jpg'))
        first_img_path = os.path.join(edited, 'piece_1_absolute.jpg')
        cv2.imwrite(first_img_path, final_puzzle)

        # combine the puzzle together and write the abs_pieces
        for i in range(1, n_images):
            transform = get_transform(matches[i - 1], is_affine)
            target_img = cv2.imread(os.path.join(pieces_pth, f'piece_{i + 1}.jpg'))

            target_img_transformed = inverse_transform_target_image(target_img, transform, final_puzzle.shape[:2][::-1])

            abs_path = os.path.join(edited, f'piece_{i+1}_absolute.jpg')
            cv2.imwrite(abs_path, target_img_transformed)

            final_puzzle = stitch(final_puzzle, target_img_transformed)
        # SAVE the final puzzle
        sol_file = f'solution.jpg'
        cv2.imwrite(os.path.join(puzzle, sol_file), final_puzzle)
