from transformers import AutoTokenizer, Qwen2ForCausalLM
import debugpy
try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9501))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass
model = Qwen2ForCausalLM.from_pretrained("lmsys/Qwen1.5-0.5B-Chat")
tokenizer = AutoTokenizer.from_pretrained("lmsys/Qwen1.5-0.5B-Chat")
prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]