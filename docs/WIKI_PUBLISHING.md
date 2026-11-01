# GitHub Wiki 发布与维护

此目录的 wiki/*.md 是 Wiki 页面源文件。**它位于代码仓库内，单独提交并不会自动改变 github.com 的 Wiki 标签页**。GitHub Wiki 存储在另一个 Git 仓库（bimanual-vla.wiki.git），需要单独推送。

## 首次发布

1. 在主仓库的 Wiki 标签页创建至少一个 Home 页面（如果已经有，跳过）。
2. 最简单的手动方式，在可访问 GitHub 的本机执行：

~~~bash
git clone https://github.com/SUNNYsyy2005/bimanual-vla.wiki.git
git clone https://github.com/SUNNYsyy2005/bimanual-vla.git
cp bimanual-vla/wiki/*.md bimanual-vla.wiki/
cd bimanual-vla.wiki
git add -- '*.md'
git commit -m "docs: publish bimanual-vla wiki"
git push origin HEAD
~~~

**注意：** 上述复制会覆盖同名的 Home / _Sidebar / _Footer 文件，但不会删除 Wiki 仓库中其他页面。推送前可先检查 git diff。系统可能要求 GitHub 凭据；不要把 token 放入脚本或提交记录。

## 可选的 GitHub Actions 同步

仓库含手动触发 workflow：.github/workflows/publish-wiki.yml。若要用它：

1. 创建具备此公开仓库 Wiki 写入权限的 GitHub 凭据，安全保存为 repository Actions secret，名称为 **WIKI_PUSH_TOKEN**（不要提交到代码库）。根据所用凭据类型检查访问范围；使用 classic PAT 时优先限制为所需最小权限（public_repo）。
2. 到 Actions → Publish GitHub Wiki → Run workflow，在 master 上运行。
3. 成功后打开 https://github.com/SUNNYsyy2005/bimanual-vla/wiki 确认 Home、侧边栏和每个页面。检查工作流日志与 git diff，如失败不要反复盲目推送。

此工作流**不会在代码提交时自动运行**，以免覆盖 Wiki 中别人的编辑。后续先编辑主仓库的 wiki/*.md，再手动同步；不要在两处同时修改同一个页面。

## 维护约定

- 页面使用英文文件名，以保证 Wiki URL 稳定；正文以中文介绍。
- 内部链接采用 GitHub Wiki 的双中括号语法；外部源码文档使用绝对 GitHub 链接。
- _Sidebar.md 和 _Footer.md 为 GitHub Wiki 专用区域；Home.md 为默认首页。
- 主仓库有更细、更常更新的开发/运维文档，Wiki 保持导览与入门性质，避免复制敏感内部环境。
- 发布时建议核查 README/代码行为是否发生变化，更新 Wiki 末尾的校对日期。
