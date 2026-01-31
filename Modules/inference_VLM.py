
def run_inference():
    with torch.no_grad():
        for batch in test_loader:
            gt_captions = batch["all_captions"]  # List[List[str]]

            generated_ids = model.generate(
                query_pixel_values=batch["query_pixel_values"],
                retrieved_pixel_values=batch["retrieved_pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=64,
                num_beams=1,
                # do_sample=True,
                # top_p=0.9,
                # temperature=0.8,
                # repetition_penalty=1.2,
            )

            decoded = T5_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for i in range(len(decoded)):
                print(gt_captions[i])
                print(decoded[i])            
            break