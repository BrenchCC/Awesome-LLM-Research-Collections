# GitHub → 飞书双语知识库自动镜像

本仓库以 GitHub `main` 为唯一真源，通过指定的飞书自建应用把论文、笔记和博客镜像到私有知识空间 `Awesome LLM Research Collections`。工作流在北京时间每天 12:00、相关内容推送到 `main` 后运行，也支持手动 `plan` 或 `apply`。

同步器只覆盖首页清单明确记录的受管节点。未进入清单的用户节点永不删除；同名冲突、重复节点、清单损坏、受管节点被移动或待删除节点含未知子节点时，同步器会停止，不猜测远端意图。

## 一次性初始化

### 1. 准备飞书应用

使用已选定的飞书自建应用，记录 App ID 和 App Secret。该应用可以复用现有应用；为应用申请并发布以下业务权限：

- `wiki:wiki`
- `docx:document`
- `docx:document.block:convert`
- `docs:document.media:upload`

当前部署复用现有应用。若应用还承担其他自动化任务，不要为了本同步器移除其已有权限；应确认上述权限已经发布，并尽可能把数据访问范围限制到目标知识空间。创建空间和添加应用成员由管理员使用用户身份完成，执行这些初始化命令的用户应用还需要相应的空间写入和成员管理权限。

### 2. 使用用户身份创建私有空间

先完成 lark-cli 用户登录，再创建空间：

```bash
lark-cli auth login
lark-cli wiki +space-create \
  --name "Awesome LLM Research Collections" \
  --description "GitHub main 的中英双语自动镜像" \
  --as user \
  --format json
```

保存返回的 `space_id`，并在飞书管理界面确认空间为私有、外部分享关闭。

### 3. 将应用加入空间

使用应用的 App ID（通常为 `cli_...`）把它添加为空间管理员：

```bash
lark-cli wiki +member-add \
  --space-id <SPACE_ID> \
  --member-id <APP_ID> \
  --member-type appid \
  --member-role admin \
  --as user \
  --format json
```

完成后，用该应用凭据执行一次只读预览。

### 4. 配置 GitHub Actions

在仓库 Settings → Secrets and variables → Actions 中创建：

- Secret `FEISHU_APP_ID`
- Secret `FEISHU_APP_SECRET`
- Variable `FEISHU_WIKI_SPACE_ID`

也可以使用 GitHub CLI 的交互式输入，避免把密钥放进命令参数或 shell 历史：

```bash
gh secret set FEISHU_APP_ID
gh secret set FEISHU_APP_SECRET
gh variable set FEISHU_WIKI_SPACE_ID
```

工作流把两个 Secret 映射为 lark-cli 官方环境变量 `LARKSUITE_CLI_APP_ID` 和 `LARKSUITE_CLI_APP_SECRET`，再通过 stdin 初始化 runner 临时目录中的 lark-cli profile。同步完成后 runner 被销毁；密钥不会进入仓库、命令参数或日志。

## 本地检查和手动同步

本地需要 lark-cli `1.0.86` 和 `rsvg-convert`。先只验证本地内容：

```bash
python scripts/sync_feishu_wiki.py --check
```

远端预览和应用使用以下环境变量：

```bash
export LARKSUITE_CLI_APP_ID="<APP_ID>"
export LARKSUITE_CLI_APP_SECRET="<APP_SECRET>"
export LARKSUITE_CLI_BRAND="feishu"
export LARKSUITE_CLI_STRICT_MODE="bot"
export FEISHU_WIKI_SPACE_ID="<SPACE_ID>"

python scripts/sync_feishu_wiki.py --plan
python scripts/sync_feishu_wiki.py --apply
```

本地未设置 `GITHUB_SHA` 时，同步器使用当前 Git HEAD。`--plan` 会读取清单、节点和 Docx revision，但不会写入；首次预览应只显示预期的 `CREATE`。`--apply` 严格串行写入，并在所有正文和安全删除完成后最后提交首页成功清单。

在 GitHub Actions 页面手动运行 **Sync Feishu Wiki** 时，先选 `plan` 检查日志，再选 `apply`。push 和定时入口固定执行 `apply`。定时 cron 为 UTC `0 4 * * *`，即北京时间 12:00；GitHub 调度可能延迟数分钟。

## 内容规则

- 论文从 `README.md` 和 `README.zh-CN.md` 的结构化条目生成，每种语言按 10 个研究分类成页。
- 笔记读取 `notes/en/` 和 `notes/zh/` 的正文 QMD；YAML front matter 被移除，callout 转为引用块，表格、代码和公式保留。
- 本地图片改写为 lark-cli 相对媒体引用；SVG 在临时目录转换为 PNG。缺失文件或单个媒体超过 20 MB 时检查失败。
- 站内 QMD 链接映射到清单中的 Wiki node token，外部链接保持不变。
- 页面哈希覆盖规范化正文和媒体字节。页面显示的 commit 只在该页正文确实更新时刷新，避免无关 Git commit 让全部页面重写。
- 飞书中的手工正文修改会造成 revision 漂移，并在下一次 `apply` 被 GitHub 真源覆盖。

飞书对 Docx 编辑和单文档写入有频率限制，因此同步器不并发写入，并只对限流与临时网络错误做最多四次指数退避。媒体上传遵守官方 20 MB 限制：

- [Docx API 概览](https://open.feishu.cn/document/server-docs/docs/docs/docx-v1/docx-overview)
- [素材上传限制](https://open.feishu.cn/document/server-docs/docs/drive-v1/media/upload_all?lang=zh-CN)

## 故障恢复

同步开始时，首页会明确写入 `in_progress` 检查点。新节点先使用包含稳定 key 哈希的确定性暂存标题创建，全部 token 记录进检查点后再原地改成最终标题；这样既避免高频改写同一个首页文档，也能在任务中断后安全识别自己创建的暂存节点。兼容旧检查点时，`pending_create_key` 也只允许认领精确父节点和精确标题。其他同名节点仍会停止同步。

建议按以下顺序恢复：

1. 重新运行 `--plan`，确认是可恢复的 `RECOVER`、revision 覆盖或剩余创建。
2. 修正权限、网络或源文件问题，不要手工删除清单 JSON。
3. 重新运行 `--apply`；成功后首页状态变为 `complete`。
4. 如果清单损坏，从飞书文档历史恢复最近一个完整首页版本，再运行 `--plan`。

删除只针对旧清单记录、但当前期望树中已不存在的节点，并按最深层优先、`include-children=false` 执行。若节点仍有任何子节点，删除会停止。不要通过编辑清单来强制删除；需要移除内容时，应先在 GitHub 真源删除并经过代码审查。

## 密钥轮换

1. 在飞书开放平台生成或重置应用密钥。
2. 立即更新 GitHub Secret `FEISHU_APP_SECRET`。
3. 手动运行 `plan`，确认应用仍能读取目标空间。
4. 手动运行 `apply`，确认首页状态和 commit 正常。
5. 撤销旧密钥，并检查 Actions 日志没有认证失败。

App ID 或目标空间变更不属于普通轮换。更换任一项时应创建新的空私有空间并执行首次 `plan`，不要把旧空间清单复制到新空间。
