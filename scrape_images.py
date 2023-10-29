import os
from bing_image_downloader import downloader


def download_bing_images(query, save_path, num_images=70):
    """
    Download images from Bing and save to the specified path.

    :param query: Search term for Bing.
    :param save_path: Path to save the downloaded images.
    :param num_images: Number of images to download (default is 70).
    """
    # Ensure the directory exists
    os.makedirs(save_path, exist_ok=True)

    # Download images from Bing
    downloader.download(query, limit=num_images, output_dir=save_path, adult_filter_off=True, force_replace=False,
                        timeout=60)


if __name__ == "__main__":
    search_and_save_paths = {
        'bee': 'C:\\My Course\\2023S2\\BeeMitesTensorFlow\\images\\bee\\bee_positive',
        'no_bee': 'C:\\My Course\\2023S2\\BeeMitesTensorFlow\\images\\bee\\bee_negative'
    }

    for term, save_path in search_and_save_paths.items():
        download_bing_images(term, save_path)

    print("Image download complete.")
