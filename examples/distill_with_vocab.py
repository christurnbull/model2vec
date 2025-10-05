from model2vec.distill import distill
from model2vec.distill.inference import PoolingMode
from transformers import AutoTokenizer


def distill_with_vocab():

    tokenizer = AutoTokenizer.from_pretrained("minishlab/potion-base-32M")
    vocab_dict = tokenizer.get_vocab()

    # IMPORTANT: Sort by token ID to maintain order
    vocab_list = [token for token, idx in sorted(
        vocab_dict.items(), key=lambda x: x[1])]

    m2v_model = distill(
        model_name="BAAI/bge-base-en-v1.5",
        pca_dims=512,
        quantize_to="float32",
        pooling=PoolingMode.MEAN,
        vocabulary=vocab_list,
        trust_remote_code=True
    )
    m2v_model.save_pretrained("m2v_bge-base-en-v1.5_d512_pmean_vpotion32m")


if __name__ == "__main__":
    distill_with_vocab()
