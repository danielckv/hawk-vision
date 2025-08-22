import os
from datetime import datetime
from time import sleep

import pandas as pd
import mesop as me


def generate_logs():
    root_dir = os.path.abspath(os.curdir)

    with open(f"{root_dir}/_log_lines.log", "r") as f:
        while True:
            for line in f:
                yield {"time": datetime.now(), "message": line}
            sleep(3)


@me.stateclass
class State:
    data_frame: pd.DataFrame = pd.DataFrame(columns=["time", "message"])


def start_streaming_from_files(action: me.ClickEvent):
    state = me.state(State)
    for val in generate_logs():
        state.data_frame = pd.concat([state.data_frame, pd.DataFrame([val])], ignore_index=True)
        yield


@me.page(
    path="/",
)
def app():
    state = me.state(State)

    with me.box(style=me.Style(padding=me.Padding.all(10))):
        me.table(
            state.data_frame,
            header=me.TableHeader(sticky=True),
            columns={
                "time": me.TableColumn(sticky=True),
                "message": me.TableColumn(sticky=True),
            },
        )

    with me.box(
            style=me.Style(
                background="#ececec",
                margin=me.Margin.all(10),
                padding=me.Padding.all(10),
            )
    ):
        me.button("Start Logger", on_click=start_streaming_from_files)
