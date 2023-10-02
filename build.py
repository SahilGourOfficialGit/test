import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from transformers import TextDataset, DataCollatorForLanguageModeling

# Check if a TPU is available, and if so, initialize it
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    print("Using TPU")
except ValueError:
    # If no TPU is available, fall back to CPU or GPU
    strategy = tf.distribute.OneDeviceStrategy(device="/CPU:0")  # You can also specify a GPU here

# Initialize a GPT-2 tokenizer and model within the strategy scope
with strategy.scope():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = TFGPT2LMHeadModel.from_pretrained("gpt2")

# Load and preprocess your training data in JSONL format
# Replace with your own data loading and preprocessing logic
# Here's a simplified example using Hugging Face's TextDataset:
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="data/small-117M-k40.train.jsonl",
    block_size=128  # Adjust block size as needed
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Define training parameters
training_args = tf.data.Dataset.from_generator

# Create a training dataset
train_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset["train"])
train_dataset = train_dataset.map(
    lambda x: tokenizer.encode(x, return_tensors="tf", truncation=True, padding="max_length"),
    num_parallel_calls=tf.data.AUTOTUNE,
)
train_dataset = train_dataset.shuffle(buffer_size=10000)
train_dataset = train_dataset.batch(batch_size=8)
train_dataset = train_dataset.map(
    lambda x: {"input_ids": x, "labels": x},
    num_parallel_calls=tf.data.AUTOTUNE,
)

# Define optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    for batch in train_dataset:
        with tf.GradientTape() as tape:
            outputs = model(batch["input_ids"], training=True)
            logits = outputs.logits
            loss_value = loss_fn(batch["labels"], logits)

        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_value:.4f}")

# Save the trained model
model.save_pretrained("small-117M-k40.train.jsonl-modelV1")
tokenizer.save_pretrained("small-117M-k40.train.jsonl-modelV1-tokenizer")
