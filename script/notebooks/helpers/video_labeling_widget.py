import os
from IPython.display import Video, display
import ipywidgets as widgets

def start_labeling(df_label, video_dir, label_file, labels):
    to_label = df_label[df_label["activity_label"].isna()]["trimmed_name"].tolist()
    video_iter = iter(to_label)

    dropdown = widgets.Dropdown(options=labels, description="Label:")
    save_button = widgets.Button(description="Save & Next", button_style="success")
    output = widgets.Output()
    current_video = [None]

    def show_next_video(_=None):
        output.clear_output(wait=True)
        try:
            vid_name = next(video_iter)
            current_video[0] = vid_name
            clip_path = os.path.join(video_dir, vid_name)
            with output:
                print(f"[STEP] Now labeling: {vid_name}")
                video_widget = Video(clip_path, embed=True, width=480)
                display(video_widget, dropdown, save_button)
        except StopIteration:
            with output:
                print("All clips labeled!")

    def save_label(_):
        vid_name = current_video[0]
        label = dropdown.value
        df_label.loc[df_label["trimmed_name"] == vid_name, "activity_label"] = label
        df_label.to_csv(label_file, index=False)
        show_next_video()

    save_button.on_click(save_label)
    display(output)
    show_next_video()