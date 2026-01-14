import os

from PIL import Image


def animate(file_list, destination_folder, prefix):
    if prefix is None:
        prefix = 'iteration'
    frames = []
    for i in file_list:
        new_frame = Image.open(i)
        frames.append(new_frame)

    # Save into a GIF file that loops forever
    try:
        os.makedirs(destination_folder)
    except FileExistsError:
        pass
    frames[0].save(os.path.join(destination_folder, f'{prefix}_animation.gif'), format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=1000, loop=0)
