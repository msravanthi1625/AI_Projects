import faiss, json
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

EMB_MODEL = "all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-base"  # choose a model appropriate to your resources

def load_index(index_path="data/index.faiss"):
    index = faiss.read_index(index_path)
    with open(index_path + ".ids.json", "r", encoding="utf-8") as f:
        ids = json.load(f)
    # load the chunks mapping (id -> text) as a single JSON object (dict)
    with open("data/processed/id2text.json", "r", encoding="utf-8") as f:
        id2text = json.load(f)  # id2text is a dict: {id: text}
    return index, ids, id2text

def retrieve(query, index, id2text, k=5):
    embed_model = SentenceTransformer(EMB_MODEL)
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    # ids is a list of chunk IDs, id2text is a dict mapping chunk IDs to text
    docs = [id2text.get(str(i), "") for i in I[0]]
    return docs

def generate_answer(query, docs):
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
    context = "\n\n".join(docs)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    out = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":
    index, ids, id2text = load_index()
    q = "What is the name of the party that announced that the winner of the leadership race will be made public on April 14, 2013, in Ottawa, Ontario? _"
    docs = retrieve(q, index, id2text, k=4)
    ans = generate_answer(q, docs)
    print(f"Question:\n {q}\n Answer:\n", ans)