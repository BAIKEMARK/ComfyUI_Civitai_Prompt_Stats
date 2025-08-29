# Civitai Prompt Stats Node

## Description

When using some models (Checkpoint / Lora), it’s often difficult to know which prompts work best.
The **Civitai Prompt Stats Node** helps you:

* Fetch **positive / negative prompts** used in community-uploaded images
* Count the most frequently used prompts and output them as a reference list
* Quickly understand commonly used prompts to optimize your own prompts

**Cache mechanism**: Results are stored in the `data` folder of the project directory to avoid redundant requests. If a model file is already cached, it will not be requested again unless `force refresh` is set to `yes`.

### Parameters

| Parameter      | Description                                                              |
| -------------- | ------------------------------------------------------------------------ |
| file\_name     | Select your local model file (CKPT or LORA)                              |
| top\_n         | Top N most frequent prompts to output (default 20)                       |
| max\_pages     | Maximum pages of images to fetch per model (default 3)                   |
| sort           | Sorting method for images: "Most Reactions" / "Most Comments" / "Newest" |
| timeout        | Request timeout in seconds (default 10)                                  |
| retries        | Number of retries for failed requests (default 2)                        |
| force\_refresh | Force refresh cache, default "no"                                        |

### How to Use

1. Place this project under ComfyUI’s `custom_nodes` directory, for example:

   ```
   ComfyUI/custom_nodes/comfyui_civitai_prompt_stats
   ```

2. Restart ComfyUI. You will see two new nodes in the node panel:

   * **Civitai Prompt Stats (CKPT)** → for Checkpoint models
   * **Civitai Prompt Stats (LORA)** → for Lora models

3. Select your local model file in the node, configure parameters, and run to get the results.

4. Outputs include positive and negative prompt lists, which can be displayed via `Show Text` nodes and are cached in the `data` folder.

5. Cached models will not trigger new requests unless `force refresh` is set to `yes`.

### Acknowledgement

This project was inspired by [Extraltodeus/LoadLoraWithTags](https://github.com/Extraltodeus/LoadLoraWithTags).


## 功能说明

在使用一些模型（Checkpoint / Lora）时，常常不知道哪些提示词最适合。
**Civitai Prompt Stats 节点**可以帮助你：

* 获取社区用户上传图片时使用的 **正向 / 负向提示词**
* 统计出现频率最高的提示词，输出为参考列表
* 快速了解别人常用的提示词，辅助你优化自己的 prompt

**缓存机制**：结果会保存在项目目录下的 `data` 文件夹中，避免重复请求。若模型文件已有缓存，默认不重复请求，除非在 `force refresh` 参数中选择 `yes`。

### 参数说明

| 参数             | 说明                                                      |
| -------------- | ------------------------------------------------------- |
| file\_name     | 选择本地模型文件（CKPT 或 LORA）                                   |
| top\_n         | 输出出现频率最高的前 N 个提示词（默认 20）                                |
| max\_pages     | 每个模型最多爬取的图片页数（默认 3）                                     |
| sort           | 图片排序方式，可选 "Most Reactions" / "Most Comments" / "Newest" |
| timeout        | 请求超时时间（秒，默认 10）                                         |
| retries        | 请求失败重试次数（默认 2）                                          |
| force\_refresh | 是否强制刷新缓存，默认 "no"                                        |

### 使用方法

1. 将项目放入 ComfyUI 的 `custom_nodes` 目录，例如：

   ```
   ComfyUI/custom_nodes/comfyui_civitai_prompt_stats
   ```

2. 重启 ComfyUI，你会在节点面板看到两个新节点：

   * **Civitai Prompt Stats (CKPT)** → 用于 Checkpoint 模型
   * **Civitai Prompt Stats (LORA)** → 用于 Lora 模型

3. 在节点中选择本地模型文件，设置参数后运行，即可得到统计结果。

4. 输出内容包括正向和负向提示词清单，可分别接入 `Show Text` 节点展示，同时缓存到 `data` 文件夹。

5. 若模型文件已有缓存，则不会重复请求，除非在 `force refresh` 中选择 `yes`。

### 鸣谢

本项目在开发过程中参考了 [Extraltodeus/LoadLoraWithTags](https://github.com/Extraltodeus/LoadLoraWithTags) 的思路，特此感谢。

---
