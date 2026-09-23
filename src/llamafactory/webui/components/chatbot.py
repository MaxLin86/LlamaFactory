# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import json
from typing import TYPE_CHECKING

from ...data import Role
from ...extras.packages import is_gradio_available
from ..bbox_plotter import plot_latest_response_boxes
from ..locales import ALERTS


if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine


QUICK_QUESTION_CHOICES = (
    ("1. 这是肠镜图像还是胃镜图像？", "这是肠镜图像还是胃镜图像？"),
    ("2. 该图像所处的消化道大类位置是什么？", "该图像所处的消化道大类位置是什么？"),
    ("3. 该图像所处的具体位置是什么？", "该图像所处的具体位置是什么？"),
    ("4. 该图像是什么光源？", "该图像是什么光源？"),
    ("5. 该图像是否有染色？", "该图像是否有染色？"),
    ("6. 该图像是否有手术帽？", "该图像是否有手术帽？"),
    ("7. 该图像中是否有器械？", "该图像中是否有器械？"),
    ("8. 该图像是否开启放大？", "该图像是否开启放大？"),
    ("9. 该图像质量如何？", "该图像质量如何？"),
    ("10. 该图像在体内还是体外？", "该图像在体内还是体外？"),
    ("11. 该病变的形态学表现是什么？", "该病变的形态学表现是什么？"),
    ("12. 该病变的病理结论是什么？", "该病变的病理结论是什么？"),
    ("13. 请进行CADX：鉴别图像中的病变，并给出诊断及相应分级。", "请进行CADX：鉴别图像中的病变，并给出诊断及相应分级。"),
    ("14. 请进行CADE：图像上是否有病变存在？若有，请指出病变位置。", "请进行CADE：图像上是否有病变存在？若有，请指出病变位置。"),
)


def check_json_schema(text: str, lang: str) -> None:
    r"""Check if the json schema is valid."""
    try:
        tools = json.loads(text)
        if tools:
            assert isinstance(tools, list)
            for tool in tools:
                if "name" not in tool:
                    raise NotImplementedError("Name not found.")
    except NotImplementedError:
        gr.Warning(ALERTS["err_tool_name"][lang])
    except Exception:
        gr.Warning(ALERTS["err_json_schema"][lang])


def create_chat_box(
    engine: "Engine", visible: bool = False
) -> tuple["Component", "Component", dict[str, "Component"]]:
    lang = engine.manager.get_elem_by_id("top.lang")
    with gr.Column(visible=visible) as chat_box:
        kwargs = {}
        if "show_copy_button" in inspect.signature(gr.Chatbot.__init__).parameters:
            kwargs["show_copy_button"] = True

        if "resizable" in inspect.signature(gr.Chatbot.__init__).parameters:
            kwargs["resizable"] = True

        chatbot = gr.Chatbot(type="messages", **kwargs)
        messages = gr.State([])
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Row():
                    with gr.Column():
                        role = gr.Dropdown(choices=[Role.USER.value, Role.OBSERVATION.value], value=Role.USER.value)
                        system = gr.Textbox(show_label=False)
                        tools = gr.Textbox(show_label=False, lines=3)

                    with gr.Column() as mm_box:
                        with gr.Tab("Image"):
                            image = gr.Image(type="pil")

                        with gr.Tab("BBox"):
                            bbox_image = gr.Image(type="pil", interactive=False)
                            bbox_info = gr.Textbox(interactive=False, lines=4)

                        with gr.Tab("Video"):
                            video = gr.Video()

                        with gr.Tab("Audio"):
                            audio = gr.Audio(type="filepath")

                        with gr.Row():
                            bbox_plot_btn = gr.Button()
                            bbox_clear_btn = gr.Button()

                quick_question = gr.Dropdown(
                    choices=QUICK_QUESTION_CHOICES,
                    value=None,
                    allow_custom_value=False,
                    filterable=True,
                )
                query = gr.Textbox(show_label=False, lines=8)
                submit_btn = gr.Button(variant="primary")

            with gr.Column(scale=1):
                max_new_tokens = gr.Slider(minimum=8, maximum=8192, value=1024, step=1)
                top_p = gr.Slider(minimum=0.01, maximum=1.0, value=0.7, step=0.01)
                temperature = gr.Slider(minimum=0.01, maximum=1.5, value=0.95, step=0.01)
                seed = gr.Number(value=-1, precision=0)
                skip_special_tokens = gr.Checkbox(value=True)
                escape_html = gr.Checkbox(value=True)
                enable_thinking = gr.Checkbox(value=True)
                clear_btn = gr.Button()

    tools.input(check_json_schema, inputs=[tools, engine.manager.get_elem_by_id("top.lang")])

    # ``input`` only reacts to a user selection. Resetting the dropdown as an
    # event output therefore allows the same preset to be selected repeatedly.
    quick_question.input(
        lambda question: (question or "", None),
        inputs=[quick_question],
        outputs=[query, quick_question],
        queue=False,
    )

    submit_btn.click(
        engine.chatter.append,
        [chatbot, messages, role, query, escape_html],
        [chatbot, messages, query],
    ).then(
        engine.chatter.stream,
        [
            chatbot,
            messages,
            lang,
            system,
            tools,
            image,
            video,
            audio,
            max_new_tokens,
            top_p,
            temperature,
            seed,
            skip_special_tokens,
            escape_html,
            enable_thinking,
        ],
        [chatbot, messages],
    )
    bbox_plot_btn.click(
        plot_latest_response_boxes,
        inputs=[image, messages, lang],
        outputs=[bbox_image, bbox_info],
    )
    bbox_clear_btn.click(lambda: (None, ""), outputs=[bbox_image, bbox_info], queue=False)
    image.change(lambda: (None, ""), outputs=[bbox_image, bbox_info], queue=False)
    clear_btn.click(lambda: ([], [], None, ""), outputs=[chatbot, messages, bbox_image, bbox_info])

    return (
        chatbot,
        messages,
        dict(
            chat_box=chat_box,
            role=role,
            system=system,
            tools=tools,
            mm_box=mm_box,
            image=image,
            bbox_image=bbox_image,
            bbox_info=bbox_info,
            bbox_plot_btn=bbox_plot_btn,
            bbox_clear_btn=bbox_clear_btn,
            video=video,
            audio=audio,
            quick_question=quick_question,
            query=query,
            submit_btn=submit_btn,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            seed=seed,
            skip_special_tokens=skip_special_tokens,
            escape_html=escape_html,
            enable_thinking=enable_thinking,
            clear_btn=clear_btn,
        ),
    )
