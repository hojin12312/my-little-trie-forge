"""Launch installed Pi without changing its installation or global settings.

Use a synthetic workspace: --workspace PATH. The MLTF backend must already
be running on loopback. Config and sessions are preserved in the printed
task-owned directory. This interactive launcher does not record a demo.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, help='optional existing synthetic workspace; otherwise create the included audited fixture')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--profile', choices=['usage','measured'], default='usage', help='usage: visible xhigh/T1/128Ki cap; measured: archived thinkingOFF/T0/2048 cap')
    parser.add_argument('--dry-run', action='store_true', help='preserve and print isolated config without sending a model request')
    parser.add_argument('--prompt', default='Read queue_cache.py and test_queue.py. Fix the bug where an older cancelled or superseded job can overwrite the result of a newer job with the same key. Add regression tests for cancellation and supersession, run python3 -m unittest -v, and report the actual result.')
    args = parser.parse_args()
    if args.workspace is None:
        workspace=Path(tempfile.mkdtemp(prefix='mltf-coding-fixture-'))/'repo'
        shutil.copytree(Path(__file__).parent/'pi_fixture',workspace)
        subprocess.run([os.sys.executable,'-m','unittest','-v'],cwd=workspace,check=True)
        subprocess.run(['git','-c','core.hooksPath=/dev/null','init','-q'],cwd=workspace,check=True)
        subprocess.run(['git','add','.'],cwd=workspace,check=True)
        subprocess.run(['git','-c','core.hooksPath=/dev/null','-c','commit.gpgsign=false','-c','user.name=fixture','-c','user.email=fixture@example.invalid','commit','-qm','audited synthetic baseline'],cwd=workspace,check=True)
        print('Preserved synthetic coding workspace:',workspace,flush=True)
    else:workspace = args.workspace.resolve(strict=True)
    if not workspace.is_dir() or not 1 <= args.port <= 65535:
        parser.error('workspace must be a directory and port must be1–65535')
    node = shutil.which('node')
    entry = Path('/opt/homebrew/lib/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js')
    if not node or not entry.is_file():
        parser.error('the validated Homebrew-arm64 Pi entrypoint and Node are required; no installation changes are made')
    root = Path(tempfile.mkdtemp(prefix='mltf-pi-isolated-'))
    config = root / 'config'; sessions = root / 'sessions'
    config.mkdir(); sessions.mkdir()
    model = {'id':'local/Qwen3.8-27B-q8c','name':'MLTF local q8c','reasoning':True,'input':['text'],
        'contextWindow':32768,'maxTokens':2048,'cost':{'input':0,'output':0,'cacheRead':0,'cacheWrite':0},
        'compat':{'supportsStore':False,'supportsDeveloperRole':False,'supportsReasoningEffort':False,
            'supportsUsageInStreaming':True,'maxTokensField':'max_tokens','thinkingFormat':'chat-template',
            'chatTemplateKwargs':{'enable_thinking':False}},
        'samplingParams':{'temperature':0,'top_p':1,'top_k':1}}
    thinking='off'
    if args.profile=='usage':
        thinking='xhigh'
        model.update(contextWindow=262144,maxTokens=131072,thinkingLevelMap={'xhigh':'xhigh','max':None})
        model['compat'].update(supportsReasoningEffort=True,chatTemplateKwargs={'enable_thinking':True,'preserve_thinking':True,'reasoning_effort':'xhigh'})
        model['samplingParams']={'temperature':1,'top_p':.95,'top_k':20,'min_p':0,'presence_penalty':0,'frequency_penalty':0,'repetition_penalty':1,'reasoning_effort':'xhigh','preserve_thinking':True}
    provider = {'baseUrl' :f'http://127.0.0.1:{args.port}/v1','api':'openai-completions','apiKey':'local-only-demo','models':[model]}
    (config/'models.json').write_text(json.dumps({'providers':{'mltf':provider}},indent=2)+'\n')
    settings = {'defaultProvider':'mltf','defaultModel':model['id'],'defaultThinkingLevel':thinking,'hideThinkingBlock':False,
        'enabledModels':['mltf/'+model['id']],'cacheWarming':'off','compaction':{'enabled':False},
        'retry':{'enabled':False},'packages':[],'extensions':[],'skills':[],'promptTemplates':[],'themes':[]}
    (config/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    (config/'auth.json').write_text('{}\n')
    environment = {k:os.environ[k] for k in ['PATH','LANG','LC_ALL','TMPDIR','TERM','COLORTERM'] if k in os.environ}
    environment.update(PI_CODING_AGENT_DIR=str(config),PI_CODING_AGENT_SESSION_DIR=str(sessions),PI_OFFLINE='1',PI_TELEMETRY='0')
    command = [node,str(entry),'--offline','--no-extensions','--no-skills','--no-prompt-templates','--no-themes',
        '--no-context-files','--provider','mltf','--model',model['id'],'--thinking',thinking,'--session-dir',str(sessions),
        '--system-prompt','Use read, bash, edit and write in this synthetic workspace. Run local tests. No network, deployment, parent-directory or user-configuration access. Report actual results.',
        '--tools','read,bash,edit,write','--',args.prompt]
    print('Preserved isolated config/session directory:',root,flush=True)
    if args.dry_run:
        print('Dry run: no Pi/model request; profile:',args.profile,flush=True)
        return
    raise SystemExit(subprocess.call(command,cwd=workspace,env=environment))

if __name__ == '__main__':
    main()
