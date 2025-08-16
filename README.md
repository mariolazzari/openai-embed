# OpenAI API: Embeddings

## Understanding embeddings

### Explore embeddings

- Numeric rappresentation of text that enables ML models to understand connections between concepts.
- Vector of strings
- Compute distance
  - small: highly related
  - large: low relatedness
  
- Information retrieval
- Sentiment analysis
- Text classification
- Document summarization

## Leverage cosine similarity

Cosine distance of the angle between two vectors.
Compute similarity between documents and queries.

- -1: complete opposite
- 0: no relationship
- 1: perfectly aligned

## Embeddings API

### Explore API

[Embeddings API](https://platform.openai.com/docs/guides/embeddings)

- Input: text
- Output: vector of values (default floats)

```js
import OpenAI from "openai";
const openai = new OpenAI();

const embedding = await openai.embeddings.create({
  model: "text-embedding-3-small",
  input: "Your text string goes here",
  encoding_format: "float",
});

console.log(embedding);
```

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [
        -0.006929283495992422,
        -0.005336422007530928,
        -4.547132266452536e-05,
        -0.024047505110502243
      ],
    }
  ],
  "model": "text-embedding-3-small",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

### Estimate pricing

Best and cheaper model: *text-embedding-ada-002*
Embeddings tokenizer: [tiktoken](https://github.com/openai/tiktoken)

```js
const { Tiktoken } = require("tiktoken/lite");
const { load } = require("tiktoken/load");
const registry = require("tiktoken/registry.json");
const models = require("tiktoken/model_to_encoding.json");

async function main() {
  const model = await load(registry[models["gpt-3.5-turbo"]]);
  const encoder = new Tiktoken(
    model.bpe_ranks,
    model.special_tokens,
    model.pat_str
  );
  const tokens = encoder.encode("hello world");
  encoder.free();
}
```

### Generate single word embeddings

Lesson 02_03 - Generate Embeddings for a Single Word

## Embeddings in real world

### Cluster similar words

Lesson 03_01 - Cluster Similar Words

### Generate embeddings for stentences

Lesson 03_02- Generate embeddings for sentences

