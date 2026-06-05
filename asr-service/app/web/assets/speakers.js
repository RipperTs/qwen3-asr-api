/* 说话人管理页（声纹库）：列表 / 改名备注 / 删除 / 降级指引。
 * 体例同 offline.js/stream.js：无构建 IIFE，依赖全局 Vue / naive / AsrCommon。
 * 仅消费 /v2/speakers* 管理 API；登记（上传样本）走 API/curl，不在本页。
 */
(function () {
  'use strict';
  const { ref, reactive, computed, watch, onMounted, h } = Vue;
  const { fmtDate, apiKey, authHeaders, mountApp, makeT, locale } = window.AsrCommon;

  const M = {
    zh: {
      'page.title': '说话人管理 - Qwen3-ASR Service',
      // 卡片头 / 刷新
      'card.title': '说话人库', 'btn.refresh': '刷新',
      // 表头与表内文案
      'col.name': '名称', 'col.source': '来源', 'col.templateCount': '模板数',
      'col.createdAt': '创建时间', 'col.actions': '操作',
      'source.auto': '自动登记', 'source.manual': '手动登记',
      'btn.edit': '改名/备注', 'btn.delete': '删除',
      'confirm.deletePositive': '删除', 'confirm.deleteNegative': '取消',
      'confirm.deleteBody': '硬删除不可恢复：声纹模板与留存音频将一并清除，该说话人在后续转写中退回匿名。',
      // 列表说明 / 空态
      'list.note': '数据永不自动清理；「自动登记」条目来自离线转写的声纹识别（identify_speakers），改名后立即在后续转写中生效。',
      'list.empty': '暂无说话人：开启 identify_speakers 转写多人音频可自动登记，或经 POST /v2/speakers 手动登记',
      // 编辑弹窗
      'edit.title': '改名 / 备注', 'edit.nameLabel': '显示名称',
      'edit.namePlaceholder': '如：张三', 'edit.noteLabel': '备注（可选）',
      'edit.notePlaceholder': '如：产品部，周会常驻',
      'btn.cancel': '取消', 'btn.save': '保存',
      // 消息提示
      'msg.nameRequired': '名称不能为空',
      'msg.saved': '已保存（后续转写将直接显示新名字）',
      'msg.saveFailed': '保存失败：{0}',
      'msg.deleted': '已删除：{0}（该说话人在后续转写中退回匿名）',
      'msg.deleteFailed': '删除失败：{0}',
      // 降级指引（四态 + error）
      'block.needKey.title': '需要 API Key',
      'block.needKey.body': '声纹库接口强制鉴权：请点击右上角钥匙图标配置 API Key 后刷新。',
      'block.unauthorized.title': 'API Key 无效',
      'block.unauthorized.body': '鉴权失败（401）：请核对右上角配置的 API Key 是否与服务端一致。',
      'block.disabled.title': '声纹库未启用',
      'block.disabled.body': '启用方法：服务端开启 enable_speaker 与 enable_speaker_db，且必须配置 api_key（声纹属生物识别信息，不允许无鉴权访问）。',
      'block.mismatch.title': '声纹模型版本不一致',
      'block.mismatch.body': '库内模板与当前引擎 model_tag 不一致：登记/识别已禁用，仅保留查看与删除。处理方式：删除库文件重建，或回退到登记时的引擎版本。',
      'block.error.title': '加载失败',
    },
    en: {
      'page.title': 'Speaker Management - Qwen3-ASR Service',
      'card.title': 'Speaker library', 'btn.refresh': 'Refresh',
      'col.name': 'Name', 'col.source': 'Source', 'col.templateCount': 'Templates',
      'col.createdAt': 'Created', 'col.actions': 'Actions',
      'source.auto': 'Auto-enrolled', 'source.manual': 'Manual',
      'btn.edit': 'Rename / Note', 'btn.delete': 'Delete',
      'confirm.deletePositive': 'Delete', 'confirm.deleteNegative': 'Cancel',
      'confirm.deleteBody': 'Hard delete is irreversible: voiceprint templates and retained audio are removed; this speaker falls back to anonymous in subsequent transcriptions.',
      'list.note': 'Data is never auto-cleaned; “Auto-enrolled” entries come from speaker identification in offline transcription (identify_speakers), and renames take effect immediately in subsequent transcriptions.',
      'list.empty': 'No speakers yet: enable identify_speakers to transcribe multi-speaker audio for auto-enrollment, or enroll manually via POST /v2/speakers',
      'edit.title': 'Rename / Note', 'edit.nameLabel': 'Display name',
      'edit.namePlaceholder': 'e.g. John', 'edit.noteLabel': 'Note (optional)',
      'edit.notePlaceholder': 'e.g. Product team, weekly regular',
      'btn.cancel': 'Cancel', 'btn.save': 'Save',
      'msg.nameRequired': 'Name cannot be empty',
      'msg.saved': 'Saved (the new name shows directly in subsequent transcriptions)',
      'msg.saveFailed': 'Save failed: {0}',
      'msg.deleted': 'Deleted: {0} (this speaker falls back to anonymous in subsequent transcriptions)',
      'msg.deleteFailed': 'Delete failed: {0}',
      'block.needKey.title': 'API Key required',
      'block.needKey.body': 'The speaker library API enforces authentication: click the key icon at the top right to configure your API Key, then refresh.',
      'block.unauthorized.title': 'Invalid API Key',
      'block.unauthorized.body': 'Authentication failed (401): verify that the API Key configured at the top right matches the server.',
      'block.disabled.title': 'Speaker library disabled',
      'block.disabled.body': 'To enable: turn on enable_speaker and enable_speaker_db on the server, and api_key must be configured (voiceprints are biometric data and may not be accessed without authentication).',
      'block.mismatch.title': 'Voiceprint model version mismatch',
      'block.mismatch.body': 'Templates in the library do not match the current engine model_tag: enrollment/identification is disabled, only view and delete remain. Fix: delete the library file and rebuild, or roll back to the engine version used at enrollment.',
      'block.error.title': 'Load failed',
    },
  };
  const t = makeT(M);

  const AppBody = {
    setup() {
      const message = naive.useMessage();

      // —— 列表状态 ——
      const rows = reactive([]);
      const loading = ref(false);
      // 降级指引：'' 正常 | need_key | unauthorized | disabled | mismatch | error
      const blocked = ref('');
      const blockedDetail = ref('');

      async function load() {
        loading.value = true;
        blocked.value = '';
        try {
          const r = await fetch('/v2/speakers', { headers: authHeaders() });
          if (r.status === 401) {
            blocked.value = apiKey.value.trim() ? 'unauthorized' : 'need_key';
            return;
          }
          if (r.status === 503) {
            const detail = (await r.json()).detail || '';
            blocked.value = detail === 'model_tag_mismatch' ? 'mismatch' : 'disabled';
            return;
          }
          if (!r.ok) {
            blocked.value = 'error';
            blockedDetail.value = 'HTTP ' + r.status;
            return;
          }
          const data = await r.json();
          rows.length = 0;
          rows.push(...(data.speakers || []));
        } catch (e) {
          blocked.value = 'error';
          blockedDetail.value = String(e);
        } finally {
          loading.value = false;
        }
      }
      onMounted(load);

      // 四态降级指引：用函数取词以随语言切换刷新（computed 内调用即响应）
      function blockGuide(key) {
        const TYPES = {
          need_key: 'warning', unauthorized: 'error',
          disabled: 'warning', mismatch: 'error', error: 'error',
        };
        const CAMEL = {
          need_key: 'needKey', unauthorized: 'unauthorized',
          disabled: 'disabled', mismatch: 'mismatch', error: 'error',
        };
        const c = CAMEL[key];
        const body = key === 'error' ? blockedDetail.value : t('block.' + c + '.body');
        return { type: TYPES[key], title: t('block.' + c + '.title'), body };
      }
      const guide = computed(() => (blocked.value ? blockGuide(blocked.value) : null));

      // —— 编辑（改名 / 备注）——
      const edit = reactive({ show: false, id: '', name: '', note: '', saving: false });
      function openEdit(row) {
        edit.id = row.id;
        edit.name = row.name;
        edit.note = row.note || '';
        edit.show = true;
      }
      async function saveEdit() {
        if (!edit.name.trim()) { message.warning(t('msg.nameRequired')); return; }
        edit.saving = true;
        try {
          const r = await fetch('/v2/speakers/' + edit.id, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json', ...authHeaders() },
            body: JSON.stringify({ name: edit.name.trim(), note: edit.note.trim() || null }),
          });
          if (!r.ok) throw new Error((await r.json()).detail || 'HTTP ' + r.status);
          message.success(t('msg.saved'));
          edit.show = false;
          await load();
        } catch (e) {
          message.error(t('msg.saveFailed', e.message));
        } finally {
          edit.saving = false;
        }
      }

      // —— 删除（硬删除，不可恢复）——
      async function removeSpeaker(row) {
        try {
          const r = await fetch('/v2/speakers/' + row.id, {
            method: 'DELETE', headers: authHeaders(),
          });
          if (!r.ok) throw new Error((await r.json()).detail || 'HTTP ' + r.status);
          message.success(t('msg.deleted', row.name));
          await load();
        } catch (e) {
          message.error(t('msg.deleteFailed', e.message));
        }
      }

      // —— 表格列（render 函数式，Naive UI data-table 体例）——
      // computed：t() 随语言切换刷新表头与表内文案
      // 英文表头/按钮明显更长（Templates / Rename / Note 等）：列宽随语言取值，避免折行
      const wide = computed(() => locale.value === 'en');
      const columns = computed(() => [
        {
          title: t('col.name'), key: 'name',
          render: row => h('div', null, [
            h('span', { style: 'font-weight:600;' }, row.name),
            row.note ? h('div', { style: 'font-size:.78em;opacity:.65;' }, row.note) : null,
          ]),
        },
        {
          title: t('col.source'), key: 'source', width: wide.value ? 132 : 96,
          render: row => h(naive.NTag, {
            size: 'small', bordered: false,
            type: row.source === 'auto' ? 'warning' : 'success',
          }, { default: () => (row.source === 'auto' ? t('source.auto') : t('source.manual')) }),
        },
        { title: t('col.templateCount'), key: 'template_count', width: wide.value ? 110 : 80, align: 'center' },
        {
          title: t('col.createdAt'), key: 'created_at', width: 170,
          render: row => fmtDate(row.created_at),
        },
        {
          title: t('col.actions'), key: 'actions', width: wide.value ? 210 : 150, align: 'right',
          render: row => h(naive.NSpace, { justify: 'end', size: 'small' }, {
            default: () => [
              h(naive.NButton, { size: 'tiny', tertiary: true, onClick: () => openEdit(row) },
                { default: () => t('btn.edit') }),
              h(naive.NPopconfirm, {
                onPositiveClick: () => removeSpeaker(row),
                positiveText: t('confirm.deletePositive'), negativeText: t('confirm.deleteNegative'),
                positiveButtonProps: { type: 'error' },
              }, {
                trigger: () => h(naive.NButton, { size: 'tiny', tertiary: true, type: 'error' },
                  { default: () => t('btn.delete') }),
                default: () => t('confirm.deleteBody'),
              }),
            ],
          }),
        },
      ]);

      // 页面标题本地化
      const setTitle = () => { document.title = t('page.title'); };
      setTitle();
      watch(locale, setTitle);

      return { rows, loading, guide, columns, load, edit, saveEdit, t };
    },
    template: `
      <div style="max-width:980px;margin:0 auto;">
        <n-card :bordered="false" class="panel" size="small">
          <template #header>
            <span class="panel-title"><a-icon name="list" size="15"></a-icon>{{ t('card.title') }}</span>
          </template>
          <template #header-extra>
            <n-button size="small" tertiary :loading="loading" @click="load">
              <a-icon name="refresh" size="14" style="margin-right:5px;"></a-icon>{{ t('btn.refresh') }}
            </n-button>
          </template>

          <n-alert v-if="guide" :type="guide.type" :title="guide.title" :show-icon="true"
                   style="margin-bottom:14px;">{{ guide.body }}</n-alert>

          <template v-if="!guide">
            <n-text depth="3" style="display:block;font-size:.8em;margin-bottom:12px;">
              {{ t('list.note') }}
            </n-text>
            <n-data-table :columns="columns" :data="rows" :loading="loading"
                          :bordered="false" size="small"
                          :row-key="row => row.id">
              <template #empty>
                <n-empty :description="t('list.empty')" size="small"></n-empty>
              </template>
            </n-data-table>
          </template>
        </n-card>

        <n-modal v-model:show="edit.show" preset="card" :title="t('edit.title')"
                 style="width:380px;" :mask-closable="!edit.saving">
          <n-space vertical size="large">
            <div>
              <n-text depth="3" style="display:block;font-size:.78em;margin-bottom:5px;">{{ t('edit.nameLabel') }}</n-text>
              <n-input v-model:value="edit.name" :placeholder="t('edit.namePlaceholder')" maxlength="64"></n-input>
            </div>
            <div>
              <n-text depth="3" style="display:block;font-size:.78em;margin-bottom:5px;">{{ t('edit.noteLabel') }}</n-text>
              <n-input v-model:value="edit.note" :placeholder="t('edit.notePlaceholder')" maxlength="200"></n-input>
            </div>
            <n-space justify="end">
              <n-button size="small" :disabled="edit.saving" @click="edit.show = false">{{ t('btn.cancel') }}</n-button>
              <n-button size="small" type="primary" :loading="edit.saving" @click="saveEdit">{{ t('btn.save') }}</n-button>
            </n-space>
          </n-space>
        </n-modal>
      </div>`,
  };

  mountApp(AppBody);
})();
