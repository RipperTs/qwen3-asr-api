# Baseline Governance

## 1. Architecture Defect

已确认的基线错误、缺口或矛盾应先修正基线，再让实现对齐修正后的基线。

## 2. Architecture Drift

实现偏离已确认且正确的基线时，应优先回归基线，不应在没有明确评审的情况下把漂移写成新基线。

## 3. Baseline Check Protocol

非平凡改动前：

1. 读取最新 baseline 快照。
2. 对比当前代码结构与所有权边界。
3. 对比当前接口、配置和持久化契约。
4. 检查是否引入新的重复所有者或调用方兜底。
5. 报告 aligned / minor drift / material drift。

## 4. Architecture Review Dimensions

1. Ownership integrity：每个职责只有一个规范所有者。
2. Module boundaries：模块边界清晰，依赖方向稳定。
3. Contract changes：接口、配置、持久化和行为变更有文档。
4. Cascade proliferation：避免新增级联补丁链。
5. Dependency direction：依赖流向稳定层。
6. Retirement completeness：旧路径要删除或明确退役条件。
7. Entropy flow：复杂度净增加必须服务明确目标。

## 5. Hard Boundaries

- 本文件是项目内 Aegis 文档工作区的治理说明。
- baseline snapshots in `baseline/` are evidence, not authority。
- ADR 或 spec 记录决策，不替代代码验证。
- This file is NEVER auto-updated；修改本文件需要显式评审。
