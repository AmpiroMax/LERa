import argparse

from moviepy.editor import CompositeVideoClip, ImageClip, TextClip, \
    concatenate_videoclips


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Video parameters.')
    parser.add_argument(
        '--split', type=str,
        choices=['valid_unseen', 'valid_seen', 'tests_seen', 'tests_unseen'],
        required=True,
        help='ALFRED data split'
    )
    parser.add_argument(
        '--run_name', type=str,
        required=True,
        help='The name of the directory inside results/[data_split]/ ' +
            'where images are saved.'
    )
    parser.add_argument(
        '--from_idx', type=int, default=0, help='Episode index to start from'
    )
    parser.add_argument(
        '--to_idx', type=int, default=120, help='Episode index to end on'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logs_folder = f'results/{args.split}/{args.run_name}'

    # TODO: better text + image composition 
    # TODO: add semantic segmentation visualization, add Checker's verdict
    # TODO: progress monitor
    for i in range(args.from_idx, args.to_idx + 1):
        ep_imgs_dir = logs_folder + f'/images/{i}'

        with open(logs_folder + f'/{i}.txt', 'r') as f:
            raw_logs = f.readlines()[2:-1]
            action_logs = [
                '0.0.0 ' + '00:00:00 ' + ' '.join(raw_logs[0].split()[2:5])
                + ' InitialAction ' + '0 ' + 'GT:True ', 
            ]  # Add initial action
            for row in raw_logs:
                if 'Checker' in row or 'Warning' in row or 'Error' in row \
                    or row[0] == ' ':
                    continue
                action_logs.append(row)

        frames = []
        for j, row in enumerate(action_logs):
            img_name = row.split()[6]
            img_clip = ImageClip(
                ep_imgs_dir + f'/{img_name}.png'
            ).set_duration(0.5)

            subgoal = ' '.join(row.split()[2:5])
            action = row.split()[5]
            # subgoal_text_clip = TextClip(txt=subgoal, color='black')
            # subgoal_text_clip = subgoal_text_clip.set_position('upper').set_duration(1)
            # action_text_clip = TextClip(txt=action, color='black')
            # action_text_clip = action_text_clip.set_position('bottom').set_duration(1)
            combined_text_clip = TextClip(
                txt=f'Subtask: {subgoal}\nAction: {action}', 
                font='Times-Bold', fontsize=12, 
                color='black'
            ).set_duration(0.5).set_position('bottom')
            
            frames.append(
                # CompositeVideoClip([img_clip, subgoal_text_clip, action_text_clip])
                CompositeVideoClip([img_clip, combined_text_clip])
            )
            # frames.append(np.array(Image.open(ep_dir + img_name)))

        video = concatenate_videoclips(frames)
        video.write_videofile(
            f'results/{args.split}/{args.run_name}/{i}.mp4', fps=2
        )
