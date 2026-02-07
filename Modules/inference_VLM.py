
def run_inference():  
    with torch.no_grad():
        for batch in test_loader:
            gt_captions = batch["all_captions"]  # List[List[str]]

            generated_ids = model.generate(
                query_pixel_values=batch["query_pixel_values"],
                retrieved_pixel_values=batch.get("retrieved_pixel_valuess"),
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=None,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.9,
                top_p=0.9,
                num_beams=4,
                early_stopping=True,
                length_penalty=1.2, # longer outputs 
                repetition_penalty=1.2 # penalize repeats
            )

            decoded = T5_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for i in range(len(decoded)):
                print(gt_captions[i][0])
                print(decoded[i])            
            break