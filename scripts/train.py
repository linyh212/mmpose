#!/usr/bin/env python3
import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train ViTPose model')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--work-dir', help='工作目录，用于保存日志和模型')
    parser.add_argument('--resume', action='store_true', help='从最新检查点恢复训练')
    parser.add_argument('--amp', action='store_true', help='使用混合精度训练')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID')
    parser.add_argument('--cfg-options', nargs='+', 
                       help='覆盖配置文件的选项，格式为 key=value')
    args = parser.parse_args()
    
    print(f"开始训练 ViTPose")
    print(f"配置文件: {args.config}")
    print(f"工作目录: {args.work_dir}")
    print(f"使用 GPU: {args.gpu_id}")
    print(f"混合精度: {args.amp}")
    print(f"恢复训练: {args.resume}")
    
    # port necessary libraries
    try:
        from mmengine.runner import Runner
        from mmengine.config import Config
    except ImportError as e:
        print(f"错误: 无法导入必要的库 - {e}")
        print("请确保已安装 MMEngine 和 MMPose")
        print("安装命令: pip install mmengine mmpose")
        sys.exit(1)
    
    # load config file
    try:
        cfg = Config.fromfile(args.config)
    except FileNotFoundError:
        print(f"错误: 找不到配置文件 {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"错误: 加载配置文件失败 - {e}")
        sys.exit(1)
    
    # set work directory
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        # use config file name as default work_dir
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        cfg.work_dir = f'work_dirs/{config_name}'
    
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    # set GPU
    cfg.gpu_ids = [args.gpu_id]
    
    # set mixed precision
    if args.amp:
        if 'optim_wrapper' not in cfg:
            cfg.optim_wrapper = {}
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
    
    # set resume
    if args.resume:
        cfg.resume = True
    
    # use cfg-options to override settings
    if args.cfg_options:
        for opt in args.cfg_options:
            key, value = opt.split('=', 1)
            keys = key.split('.')
            current = cfg
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # type conversion
            try:
               value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
            
            current[keys[-1]] = value
    
    # print final config
    print("\n最终配置:")
    print(f"工作目录: {cfg.work_dir}")
    print(f"使用 GPU: {cfg.gpu_ids}")
    print(f"训练轮数: {cfg.train_cfg.max_epochs}")
    print(f"Batch size: {cfg.train_dataloader.batch_size}")
    
    # build and run the runner
    try:
        runner = Runner.from_cfg(cfg)
        print("\n开始训练...")
        runner.train()
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()