# oneft.py
import os, json, sys, time, gc
import torch
from datasets import DatasetDict
from finetuning import train_final_model
from finetune import load_labelintdcts

if __name__ == '__main__':
    cwd = os.getcwd()
    int2label_dct, label2int_dct = load_labelintdcts()
    #output_dir = "E:/PhD/2June2025/"
    output_dir = cwd+"/outputs/"
    input_dir = cwd+"/inputs/"
    torch.cuda.empty_cache()
    gc.collect()
    r = 9
    for mode in ["bn", "mc"]:
        while not torch.cuda.is_available():
            print("Cuda unavailable")
            time.sleep(3)
        print("\nCuda freed!")
        st = time.time()
        print(f"\n--- Starting {mode} run ---")
        print("Start", torch.cuda.memory_allocated())
        modeldir = os.path.join(output_dir, f"final_{mode}_model")
        os.makedirs(modeldir, exist_ok=True)
        i = 0
        while not torch.cuda.is_available():
            time.sleep(3)
            i+=1
            if i>5:
                sys.exit()
        ds = DatasetDict.load_from_disk(f"{input_dir}/ds_{mode}")
        metrics = train_final_model(
            ds,
            int2label_dct[mode],
            label2int_dct[mode],
            mode,
            dev='cuda',
            output_dir=output_dir
        )
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(3)

        print(f"\n--- Finished {mode} run ---")
        print(f'\nDone in {round((time.time()-st)/60,2)} min')
        print("End", torch.cuda.memory_allocated())
        time.sleep(2)