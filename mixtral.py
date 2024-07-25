import ollama
import tqdm

with open("chiralb.txt") as f:
    lines = f.readlines()

model = 'mixtral:8x7b'

first_prompt = '''TThe following text is a medieval medical treatise written in Old French, around the 12th-13th centuries.
Translate it line by line into English like this:
When chronic pain comes in the whole head and lasts a long time, the patient should use salves and pills called "cochie", and purgatives for the head and oils and plasters; and if the cautery we mentioned before is applied and it does not work.
Here comes the first sentence to translate. Do not add any comments or contextual information about the text, like "Here is the translation", don't provide any comments about the text like, "Wow this is hard!", etc. Do not comment on whether the translation is accurate or not, it doesn't matter. Just translate the text line-by-line into English and that should be the only text produced in the final output file and nothing else:
'''

ctx = 8192

for fnum in range(0, 10):
    outfile = open(f"chiralb-mixtral.txt", "w")

    response = ollama.generate(model=model, prompt=first_prompt + lines[0], options={"num_ctx": ctx})
    print(f"{lines[0].strip()}\t{response['response'].splitlines()[0].strip()}", file=outfile)

    for line in tqdm.tqdm(lines[1:]):
        print(len(response['context']))
        if len(response['context']) > 7000:
            response = ollama.generate(model=model, prompt=first_prompt + line, options={"num_ctx": ctx})
        else:
            response = ollama.generate(model=model, prompt=line, options={"num_ctx": ctx}, context=response['context'])
        if response['response'].strip() == "" or 'context' not in response:
            response = ollama.generate(model=model, prompt=first_prompt + line, options={"num_ctx": ctx})
        if response['response'].strip() == "":
            print(f"{line.strip()}\t", file=outfile)
        else:
            print(f"{line.strip()}\t{response['response'].splitlines()[0].strip()}", file=outfile)

    outfile.close()
