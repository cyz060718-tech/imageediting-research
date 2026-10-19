"""Local debugging UI. Run with .venv/Scripts/python.exe app.py."""
import bootstrap  # noqa: F401
import json
import os
import time
import uuid
from pathlib import Path

# Gradio reads these defaults during import; leaving server_port unset enables retries.
os.environ.setdefault('GRADIO_SERVER_PORT', '7861')
os.environ.setdefault('GRADIO_NUM_PORTS', '100')
import gradio as gr
import numpy as np
import config
import dataset
from core import editor, masker, metrics
from core.instruction_parser import parse
from eval import write_json

EXAMPLES=[[str(dataset.resolve(it['image'])),it.get('example_instruction_zh') or it['instruction']]
          for it in dataset.load_items() if it['split']=='dev']

def select_manual(image, manual, mask_mode):
    if mask_mode == 'auto':
        return None
    if mask_mode != 'manual':
        raise ValueError('请选择自动识别或人工选区。')
    if manual is None:
        raise ValueError('请先手绘或上传黑白选区；也可以切回自动识别。')
    return masker.validate_manual_mask(manual, source=image)

def show_mask_mode(mode):
    return (gr.update(visible=mode=='manual'), gr.update(visible=mode=='auto'), *clear_results())

def clear_results():
    return None,None,None,None,None

def reset_image():
    return None,None,'auto',*clear_results()

def load_paint(image):
    if image is None:raise gr.Error('请先上传原图。')
    from PIL import ImageOps
    base=ImageOps.exif_transpose(image).convert('RGBA')
    return {'background':base,'layers':[],'composite':base}

def use_paint(image,canvas):
    from PIL import Image,ImageOps
    if image is None or not canvas or not canvas.get('background'):
        raise gr.Error('先载入原图，再涂画要编辑的区域。')
    base=ImageOps.exif_transpose(image).convert('RGB')
    background=canvas['background'].convert('RGB')
    if base.size!=background.size or not np.array_equal(np.asarray(base),np.asarray(background)):
        raise gr.Error('原图已经变化，请重新载入原图绘制选区。')
    alpha=np.zeros((base.height,base.width),np.uint8)
    for layer in canvas.get('layers',[]):
        layer=layer.convert('RGBA')
        if layer.size!=base.size:raise gr.Error('手绘图层尺寸不一致，请重新载入原图。')
        alpha=np.maximum(alpha,np.asarray(layer)[...,3])
    if not np.any(alpha>127):raise gr.Error('还没有绘制选区，请用画笔涂抹要修改的区域。')
    return Image.fromarray((alpha>127).astype(np.uint8)*255)

def preview(image,instruction,manual,threshold,dilate,feather,llm,method='quality',mask_backend=config.DEFAULT_MASK_BACKEND,mask_mode='auto'):
    if image is None:raise gr.Error('请上传原图。')
    try:
        manual=select_manual(image,manual,mask_mode)
        intent=parse(instruction,llm);source=editor.prepare_for_method(image,intent,method)
        bundle=editor.get_bundle(source,intent,manual,threshold,int(dilate),int(feather),backend=mask_backend)
        if method=='quality' and intent.action=='color':
            from core.appearance import recolor
            _,alpha,_=recolor(source,bundle['raw'],intent.new_content)
            bundle=dict(bundle,hard=(alpha>0).astype(np.uint8)*255)
        info={'intent':intent.to_dict(),'mask_mode':mask_mode,'mask_source':bundle['source'],
              'area_fraction':bundle['area_fraction'],'selected_regions':bundle['component_count'],
              'region_definition':'detected_instances' if bundle['source']=='grounding_dino_sam' else 'connected_regions',
              'note':'请确认红色区域选中了正确对象；定位正确仍不保证重绘数量和外观正确。'}
        return masker.overlay(source,bundle['hard']),info
    except Exception as exc:raise gr.Error(str(exc))

def run(image,instruction,manual,method,steps,guidance,strength,seed,threshold,dilate,feather,llm,mask_backend=config.DEFAULT_MASK_BACKEND,mask_mode='auto'):
    if image is None:raise gr.Error('请上传原图。')
    try:
        manual=select_manual(image,manual,mask_mode)
        result=editor.edit(image,instruction,method,steps=int(steps),guidance=guidance,
            strength=None if strength==0 else strength,seed=int(seed),use_llm=llm,
            mask_threshold=threshold,dilate=int(dilate),feather=int(feather),manual_mask=manual,mask_backend=mask_backend)
        out=config.OUTPUT_DIR/'interactive'/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        out.mkdir(parents=True)
        result['input'].save(out/'input.png');result['result'].save(out/'result.png');result['raw'].save(out/'raw.png')
        if result['mask'] is not None:
            from PIL import Image
            Image.fromarray(result['mask']).save(out/'mask_hard.png')
        if result['bundle'] is not None:
            np.save(out/'soft_mask.npy',result['bundle']['soft'])
            from PIL import Image
            Image.fromarray(result['bundle']['raw']).save(out/'mask_raw.png')
        if manual is not None:
            manual.save(out/'manual_mask.png')
        if result['mask_overlay'] is not None:
            result['mask_overlay'].save(out/'mask_overlay.png')
        if result.get('candidates'):
            candidate_dir=out/'candidates';candidate_dir.mkdir()
            for index,candidate in enumerate(result['candidates']):
                candidate.save(candidate_dir/f'attempt_{index}.png')
        record={'instruction':instruction,'intent':result['intent'].to_dict(),'method':method,
                'steps':steps,'guidance':guidance,'strength':result['strength'],'seed':seed,
                'threshold':threshold,'dilate':dilate,'feather':feather,'time':result['time'],
                'peak_vram_mb':result['peak_vram_mb'],'device':result['device'],
                'engine':result.get('engine'),'engine_info':result.get('engine_info'),
                'effective_steps':result.get('effective_steps'),'raw_kind':result.get('raw_kind'),
                'mask_mode':mask_mode,'mask_backend':mask_backend if manual is None else 'manual',
                'mask_source':result['bundle']['source'] if result.get('bundle') else None,
                'evaluation':'尚未提供独立 GT，不报告背景保持率或 Mask IoU。',
                'output_directory':str(out)}
        write_json(out/'record.json',record)
        check=(result.get('engine_info') or {}).get('automatic_check',{})
        if check.get('status')=='needs_review':
            gr.Warning(f"有 {len(check['unconfirmed_regions'])} 个目标未能自动确认替换成功，请检查对象数量和外观。")
        raw_label='颜色变换结果' if result.get('engine')=='lab_chroma' else ('逐对象结果' if result.get('raw_kind') in {'sequential_composites','local_instance_composites'} else '融合前')
        return result['result'],result['mask_overlay'],[(result['input'],'原图'),(result['raw'],raw_label),
                (result['result'],'最终结果')],record,str(out/'result.png')
    except Exception as exc:raise gr.Error(str(exc))

def build_demo():
    with gr.Blocks(title='图像局部编辑 · 调试改进版') as demo:
        gr.Markdown('# 图像局部编辑\n上传原图并输入指令，即可自动识别编辑对象。改色保留原图纹理，删除修补背景，替换逐个对象重绘。')
        with gr.Tab('编辑与排错'):
            with gr.Row():
                with gr.Column():
                    image=gr.Image(type='pil',format='png',label='原图',height=300)
                    instruction=gr.Textbox(label='编辑指令',value='把外套换成蓝色')
                    with gr.Accordion('选择示例图片',open=False):
                        gr.Examples(EXAMPLES,inputs=[image,instruction],examples_per_page=6)
                    mask_mode=gr.Radio([('自动识别编辑对象','auto'),('手绘或上传黑白选区','manual')],
                        value='auto',label='编辑区域')
                    with gr.Group(visible=False) as manual_controls:
                        gr.Markdown('只在自动区域不准确时使用。**原照片只放在上面的「原图」中。**')
                        with gr.Accordion('手绘选区',open=True):
                            paint_load=gr.Button('载入原图，开始涂画')
                            paint=gr.ImageEditor(type='pil',image_mode='RGBA',sources=[],format='png',
                                transforms=[],layers=False,brush=gr.Brush(colors=['#ff4040'],color_mode='fixed'),
                                label='涂抹需要编辑的部分，橡皮擦可擦除',height=350)
                            paint_apply=gr.Button('使用手绘选区')
                        manual=gr.Image(type='pil',image_mode='RGB',format='png',
                            label='选区图：白色编辑，黑色保留（也可上传同尺寸黑白掩码）')
                    method=gr.Dropdown([(v,k) for k,v in editor.METHODS.items()],value='quality',label='编辑方式')
                    backend=gr.Dropdown([('精细实例分割（DINO + SAM）','grounded_sam'),('原版快速分割（CLIPSeg）','clipseg')],
                        value=config.DEFAULT_MASK_BACKEND,label='自动选区方式')
                    with gr.Accordion('调试参数',open=False):
                        steps=gr.Slider(10,50,config.DEFAULT_STEPS,step=1,label='每次重绘步数（保真改色/删除不使用）')
                        guidance=gr.Slider(1,15,config.DEFAULT_GUIDANCE,step=.5,label='文本引导强度（保真改色/删除不使用）')
                        strength=gr.Slider(0,1,0,step=.05,label='重绘强度：0 使用默认 1.0（保真改色/删除不使用）')
                        threshold=gr.Slider(.1,.9,.4,step=.05,label='分割阈值')
                        dilate=gr.Slider(0,31,config.MASK_DILATE,step=2,label='选区扩张核尺寸（替换/删除）')
                        feather=gr.Slider(0,31,config.MASK_FEATHER,step=2,label='边缘融合核尺寸（替换/删除）')
                        seed=gr.Number(42,precision=0,label='随机种子')
                        llm=gr.Checkbox(False,label='使用已配置的 LLM 解析接口')
                    preview_button=gr.Button('预览掩码')
                    run_button=gr.Button('开始编辑',variant='primary')
                with gr.Column():
                    edited=gr.Image(type='pil',format='png',label='编辑结果',height=350)
                    overlay=gr.Image(type='pil',format='png',label='编辑区域预览',height=250)
                    download=gr.File(label='下载结果')
                    with gr.Accordion('对比原图和编辑结果',open=False):
                        gallery=gr.Gallery(label='原图 / 融合前 / 最终结果',columns=3)
                    with gr.Accordion('查看运行详情',open=False):
                        info=gr.JSON(label='解析、参数与运行记录')
            result_outputs=[edited,overlay,gallery,info,download]
            preview_button.click(preview,[image,instruction,manual,threshold,dilate,feather,llm,method,backend,mask_mode],[overlay,info],
                                 concurrency_limit=1,concurrency_id='image_editing')
            run_button.click(run,[image,instruction,manual,method,steps,guidance,strength,seed,threshold,dilate,feather,llm,backend,mask_mode],
                             [edited,overlay,gallery,info,download],concurrency_limit=1,concurrency_id='image_editing')
            paint_load.click(load_paint,[image],[paint])
            paint_apply.click(use_paint,[image,paint],[manual])
            mask_mode.change(show_mask_mode,[mask_mode],[manual_controls,backend,*result_outputs],queue=False)
            image.change(reset_image,[],[manual,paint,mask_mode,*result_outputs],queue=False)
            for control in (instruction,method,backend,manual):
                control.change(clear_results,[],result_outputs,queue=False)
        with gr.Tab('数据与实验'):
            report=dataset.validate(split='dev')
            test_count=sum(it['split']=='test' for it in dataset.load_items())
            gr.Markdown('已登记调试样本：**'+str(report['registered'])+'**。已登记预留测试样本：**'+str(test_count)+'**，目标为改色、删除、替换各 40 张。\n\n'
                'COCO 样本使用官方提供的独立人工实例标注作为 GT，并保留来源和许可证。原有 3 张演示图尚缺 GT。\n\n'
                '未标注时，背景指标和 IoU 留空。参数只在调试集调整，正式测试前冻结。')
            catalog=config.ROOT/'data/dataset_catalog.csv'
            gr.File(value=str(catalog if catalog.exists() else config.ROOT/'data/collection_plan.csv'),label='数据目录与标注来源',interactive=False)
            gr.Markdown('```text\npython dataset.py validate --split dev\npython eval.py --split dev\npython eval.py --split dev --methods ours ours_instances --seeds 42 123 2026\npython dataset.py freeze --settings experiment.json\npython eval.py --split test\n```\n'
                '实验输出包括逐样本结果、原始生成图、掩码、指标明细、失败记录和人工评价表。')
        with gr.Tab('判定标准'):
            gr.Markdown('**改色**：目标颜色正确，衣服款式、纹理、脸部与姿态尽量保留。\n\n'
                '**删除**：先定义是否包含支撑杆及附属部分，再检查完整删除和背景填补。\n\n'
                '**替换**：检查新对象类别、数量、位置和比例。\n\n'
                '失败类型：parser_error / localization_error / semantic_failure / count_error / '
                'identity_drift / background_drift / boundary_artifact / runtime_error。\n\n'
                '连通域逐个编辑是实验选项，接触或重叠的物体仍可能合并。降低改色强度与修改提示词的收益需要实测。')
    return demo

if __name__=='__main__':
    build_demo().queue(default_concurrency_limit=1).launch(server_name='127.0.0.1',inbrowser=True)
