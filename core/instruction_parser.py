"""模块一：自然语言指令解析（Natural Language Interpreter）

把用户的自然语言指令拆解为结构化编辑意图：
    {action, target, new_content, edit_prompt, mask_prompt}

- action:      replace / remove / add / color / style
- target:      要编辑的物体（英文，送给分割模型生成掩码）
- new_content: 编辑后的内容（英文，送给扩散模型）
- edit_prompt: 补全光影/质感等描述后的最终生成提示词

两种实现：
1. 规则解析器（默认，离线可用）：正则模板 + 中英词表
2. LLM 解析器（可选）：调用 OpenAI 兼容接口，配置 LLM_API_KEY 后启用
"""
import json
import re
from dataclasses import dataclass, asdict

import config

# 常用中文词 -> 英文（够演示用，可自行扩充）
ZH_EN = {
    # 物体
    "猫": "cat", "狗": "dog", "人": "person", "男人": "man", "女人": "woman", "女孩": "girl", "男孩": "boy",
    "模特": "model", "衣服": "clothes", "上衣": "shirt", "衬衫": "shirt", "外套": "jacket", "裙子": "dress",
    "裤子": "pants", "帽子": "hat", "眼镜": "glasses", "鞋": "shoes", "鞋子": "shoes", "包": "bag",
    "头发": "hair", "脸": "face", "天空": "sky", "云": "clouds", "草地": "grass", "草": "grass", "树": "tree",
    "花": "flowers", "山": "mountain", "水": "water", "海": "sea", "河": "river", "路": "road", "路灯": "street lamp",
    "车": "car", "汽车": "car", "自行车": "bicycle", "房子": "house", "建筑": "building", "窗户": "window",
    "门": "door", "桌子": "table", "椅子": "chair", "沙发": "sofa", "杯子": "cup", "瓶子": "bottle",
    "苹果": "apple", "香蕉": "banana", "蛋糕": "cake", "背景": "background", "墙": "wall", "地面": "ground",
    "雪": "snow", "石头": "stone", "鸟": "bird", "马": "horse", "老虎": "tiger", "狮子": "lion", "兔子": "rabbit",
    "熊": "bear", "西装": "suit", "职业装": "business suit", "领带": "tie", "雨伞": "umbrella", "手机": "phone",
    "停车标志": "stop sign", "标志": "sign", "牌子": "sign", "电脑": "laptop", "书": "book", "球": "ball", "船": "boat", "飞机": "airplane", "月亮": "moon", "太阳": "sun",
    # 颜色
    "红色": "red", "蓝色": "blue", "绿色": "green", "黄色": "yellow", "黑色": "black", "白色": "white",
    "紫色": "purple", "粉色": "pink", "橙色": "orange", "灰色": "gray", "棕色": "brown", "金色": "golden",
    "红": "red", "蓝": "blue", "绿": "green", "黄": "yellow", "黑": "black", "白": "white", "紫": "purple",
    # 风格/属性
    "夕阳": "sunset", "夜晚": "night", "冬天": "winter", "秋天": "autumn", "春天": "spring", "夏天": "summer",
    "油画": "oil painting", "水彩": "watercolor", "卡通": "cartoon", "素描": "sketch", "赛博朋克": "cyberpunk",
    "木头": "wooden", "木质": "wooden", "金属": "metal", "玻璃": "glass", "一只": "a", "一个": "a", "一件": "a",
    "一顶": "a", "一副": "a pair of", "一辆": "a", "一朵": "a", "一些": "some",
}

COLORS_EN = {v for k, v in ZH_EN.items() if k.endswith("色")}
CLOTHES_EN = ["clothes", "shirt", "jacket", "dress", "suit", "pants", "coat"]

QUALITY_WORDS = "high quality, photorealistic, natural lighting, detailed texture, consistent with surroundings"
NEGATIVE_PROMPT = "blurry, low quality, distorted, deformed, artifacts, watermark, text, bad anatomy"


@dataclass
class EditIntent:
    action: str
    target: str
    new_content: str
    edit_prompt: str
    mask_prompt: str
    negative_prompt: str = NEGATIVE_PROMPT
    parser: str = "rule"

    def to_dict(self):
        return asdict(self)


def translate(text: str) -> str:
    """简易中译英：按词表最长匹配替换，英文原样保留。"""
    text = text.strip().strip("。.!！ ")
    if not re.search(r"[一-鿿]", text):
        return text
    keys = sorted(ZH_EN, key=len, reverse=True)
    out, i = [], 0
    while i < len(text):
        for k in keys:
            if text.startswith(k, i):
                out.append(ZH_EN[k])
                i += len(k)
                break
        else:
            ch = text[i]
            if not re.match(r"[一-鿿]", ch):
                out.append(ch)
            i += 1  # 词表外的中文字直接跳过（如“的”“了”）
    return re.sub(r"\s+", " ", " ".join(out)).strip()


# (正则, action)；组名 target / new
ZH_PATTERNS = [
    (r"把(?P<target>.+?)(?:换成|变成|改成|替换成|替换为|改为)(?P<new>.+)", "replace"),
    (r"将(?P<target>.+?)(?:换成|变成|改成|替换成|替换为|改为)(?P<new>.+)", "replace"),
    (r"(?:去掉|删除|移除|删掉|擦除|抹掉)(?P<target>.+)", "remove"),
    (r"把(?P<target>.+?)(?:去掉|删除|移除|删掉|擦除|抹掉)", "remove"),
    (r"给(?P<target>.+?)换(?:上|一件|一身|一套)?(?P<new>.+)", "wear"),
    (r"(?:给|在)(?P<target>.+?)(?:加上|戴上|添加|加|放)(?P<new>.+)", "add"),
    (r"把(?P<target>.+?)(?:染成|涂成|改成)(?P<new>.+?色)", "color"),
    (r"(?P<target>.+?)(?:变成|换成)(?P<new>.+)", "replace"),
]
EN_PATTERNS = [
    (r"(?:replace|change|turn|make)\s+(?:the\s+)?(?P<target>.+?)\s+(?:with|to|into)\s+(?P<new>.+)", "replace"),
    (r"(?:remove|delete|erase)\s+(?:the\s+)?(?P<target>.+)", "remove"),
    (r"(?:add|put)\s+(?P<new>.+?)\s+(?:on|to|in)\s+(?:the\s+)?(?P<target>.+)", "add"),
]


def _build(action: str, target_en: str, new_en: str, parser: str) -> EditIntent:
    if action == "remove":
        new_en = "empty background"
        prompt = f"clean natural background, nothing there, {QUALITY_WORDS}"
    elif action == "add":
        prompt = f"{target_en} with {new_en}, {QUALITY_WORDS}"
    elif action == "color":
        prompt = f"{new_en} {target_en}, {QUALITY_WORDS}"
    else:
        prompt = f"a photo of {new_en}, full body, {QUALITY_WORDS}"
    negative = NEGATIVE_PROMPT
    if action != "remove":
        # 防止 Inpainting 把掩码区域直接填成背景、导致目标“消失”
        negative += ", empty, nothing, plain background"
    return EditIntent(action, target_en, new_en, prompt, target_en, negative, parser=parser)


def rule_parse(instruction: str) -> EditIntent:
    s = instruction.strip().strip("。.!！ ")
    if not s:
        raise ValueError("请输入编辑指令。")
    is_zh = bool(re.search(r"[一-鿿]", s))
    for pat, action in (ZH_PATTERNS if is_zh else EN_PATTERNS):
        m = re.search(pat, s, flags=re.I)
        if m:
            raw_target = m.group("target")
            if is_zh:  # “背景里的路灯”“图中的猫” -> 只保留中心词
                # Strip known context phrases; splitting at 上 broke 上衣.
                raw_target = re.sub(r"^(?:图片中|图中|照片中|背景里|背景中|画面中)(?:的)?", "", raw_target)
            target = translate(raw_target)
            new = translate(m.group("new")) if "new" in m.groupdict() and m.group("new") else ""
            if action == "replace" and new in COLORS_EN:
                action = "color"  # “把衣服换成红色” -> 改颜色
            if action == "wear":  # “给模特换一件蓝色职业装” -> 掩码对准衣服而不是人
                action = "replace"
                target = "clothes" if any(w in new for w in CLOTHES_EN) else target
            return _build(action, target, new, "rule")
    # 兜底：整句作为提示词，掩码目标未知
    text = translate(s)
    return EditIntent("global", "", text, f"{text}, {QUALITY_WORDS}", "", parser="rule")


LLM_SYSTEM = """你是图像编辑指令解析器。把用户指令解析为 JSON，字段全部用英文：
{"action": "replace|remove|add|color|style", "target": "要编辑的物体，简短名词短语",
 "new_content": "编辑后的内容", "edit_prompt": "给 Stable Diffusion 的详细英文提示词，补充光影、质感描述"}
只输出 JSON。"""


def llm_parse(instruction: str) -> EditIntent:
    from openai import OpenAI
    client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "system", "content": LLM_SYSTEM}, {"role": "user", "content": instruction}],
        temperature=0.2,
    )
    raw = resp.choices[0].message.content.strip().strip("`")
    raw = raw[raw.find("{"): raw.rfind("}") + 1]
    d = json.loads(raw)
    return EditIntent(d["action"], d["target"], d.get("new_content", ""), d["edit_prompt"], d["target"], parser="llm")


def parse(instruction: str, use_llm: bool = False) -> EditIntent:
    if use_llm and config.LLM_API_KEY:
        try:
            return llm_parse(instruction)
        except Exception as e:  # LLM 不可用时退回规则解析
            print(f"[parser] LLM 解析失败，使用规则解析: {e}")
    return rule_parse(instruction)


if __name__ == "__main__":
    for t in ["把猫换成老虎", "去掉背景里的路灯", "给模特换一件蓝色的职业装", "把衣服换成红色",
              "replace the cat with a dog", "remove the car"]:
        print(t, "->", rule_parse(t).to_dict())
