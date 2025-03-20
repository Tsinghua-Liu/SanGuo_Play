import gradio as gr
from PIL import Image
import base64
from typing import List, Tuple
from pathlib import Path

def create_chat_interface(history: List[Tuple[str, str]], max_visible: int = 3) -> gr.HTML:
    """
    创建带反向滚动功能的聊天界面
    :param history: 聊天历史列表，每个元素为 (base64_img, text)
    :param max_visible: 默认可见消息条数
    :return: gradio.HTML组件
    """
    messages_html = []

    # 新增空状态处理
    if not history:
        messages_html.append("""
        <div style="
            text-align: center;
            color: #666;
            padding: 20px;
            font-style: italic;
        ">
            等待对话开始...
        </div>
        """)


    for idx, (img_base64, text) in enumerate(history[::-1]):
        message = f"""
        <div class="message-item" style="
            margin: 10px 0;
            padding: 12px;
            background: #f8f9fa;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        ">
            <div style="
                display: flex;
                align-items: flex-start;
                gap: 12px;
            ">
                <img src="data:image/png;base64,{img_base64}" 
                     style="
                         width: 50px;
                         height: 50px;
                         border-radius: 6px;
                         flex-shrink: 0;
                         object-fit: cover;
                     ">
                <div style="
                    color: #111; /* 设置字体颜色 */
                    font-size: 20px; /* 设置字体大小 */
                    line-height: 2.0;
                    word-wrap: break-word;
                    max-width: 900px;
                ">
                    {text}
                </div>
            </div>
        </div>
        """
        messages_html.append(message)

    html_content = f"""
    <div id="chat-container" style="
        height: 800px;
        overflow-y: auto;
        padding: 15px;
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid #e9ecef;
        display: flex;
        flex-direction: column-reverse;  /* 关键：反向排列 */
    ">
        {"".join(messages_html)}
    </div>
    <script>
        // 自动滚动到底部（实际为物理顶部）
        const container = document.getElementById('chat-container');
        container.scrollTop = 0;  // 反向容器需要滚动到顶部

        // 监听容器变化保持位置
        const observer = new MutationObserver(() => {{
            container.scrollTop = 0;
        }});
        observer.observe(container, {{ childList: true, subtree: true }});
    </script>
    """
    return html_content

# blockscss = """
# .progress { display: none; }
# .message-item:hover {
#     background: #f1f3f5 !important;
#     transition: background 0.2s ease;
# }
# """

blockscss = """
/* 为聊天消息添加悬停效果 */
.message-item:hover {
    background: #f1f3f5 !important;
    transition: background 0.2s ease;
}
"""

