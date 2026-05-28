import equinox as eqx
import jax
import jax.numpy as jnp
from functools import partial


# the choice for equinox is ease of use with jax
# https://www.reddit.com/r/MachineLearning/comments/u34oh2/d_what_jax_nn_library_to_use/
# https://docs.kidger.site/equinox/
class NeuralNetwork(eqx.Module):
    layers: list

    def __init__(self, n_features, last_layer_scale=5.0):
        key = jax.random.PRNGKey(seed=0) # Make this configurable?!
        key1, key2, key3 = jax.random.split(key, 3)
        last_linear = eqx.nn.Linear(100, 1, key=key3)
        # scale up last layer so initial sigmoid outputs span [0.1, 0.9] instead
        # of clustering near 0.5 — avoids the CLs cold-start weak-gradient problem
        last_linear = eqx.tree_at(
            lambda l: l.weight, last_linear, last_linear.weight * last_layer_scale
        )
        self.layers = [
            eqx.nn.Linear(n_features, 100, key=key1),
            jax.nn.relu,
            eqx.nn.Linear(100, 100, key=key2),
            jax.nn.relu,
            last_linear,
            jax.nn.sigmoid,
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# this is not optimized and adds a multihead attention, but works
# interchangably to NeuralNetwork 
class NeuralNetworkFeatureAttention(eqx.Module):
    # This layer maps a scalar (1 value) to an embedding of size `embed_dim`.
    feature_embed: eqx.nn.Linear
    # Equinox’s built-in multi-head attention.
    attn: eqx.nn.MultiheadAttention
    # Final classifier: takes the flattened attended tokens (n_features * embed_dim).
    classifier: eqx.nn.Linear

    def __init__(
        self,
        n_features: int,
        embed_dim: int = 32,
        num_heads: int = 1,
        key: "jax.random.PRNGKey" = jax.random.PRNGKey(0),
    ):
        # embed_dim = 2 * n_features
        k1, k2, k3 = jax.random.split(key, 3)
        # Map each scalar feature to a vector of length `embed_dim`.
        self.feature_embed = eqx.nn.Linear(1, embed_dim, key=k1)
        # IMPORTANT: Pass parameters as keywords so that num_heads isn’t misinterpreted.
        self.attn = eqx.nn.MultiheadAttention(
            query_size=embed_dim, num_heads=num_heads, key=k2
        )
        # The classifier receives a flattened output: (n_features * embed_dim,)
        self.classifier = eqx.nn.Linear(n_features * embed_dim, 1, key=k3)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Process a single event.

        Args:
            x: jnp.ndarray of shape (n_features,) representing one event’s feature vector.

        Returns:
            A scalar (after sigmoid) representing the event’s prediction.
        """
        # Each event is a vector (n_features,). We “tokenize” it by reshaping each scalar into a token.
        # x_tokens becomes (n_features, 1)
        x_tokens = x[:, None]
        # Apply the feature embedding to each token independently.
        # We use vmap so that each token of shape (1,) is processed as expected.
        # Result: (n_features, embed_dim)
        x_emb = jax.vmap(self.feature_embed)(x_tokens)
        # Use self-attention on the sequence of feature embeddings.
        # We use the same tensor for query, key, and value so that each feature attends to every other.
        # Equinox’s MultiheadAttention here expects a 2D tensor (sequence_length, embed_dim).
        x_att = self.attn(x_emb, x_emb, x_emb)
        # Flatten the attended output into a 1D vector.
        x_flat = jnp.ravel(x_att)
        # Classify and squash the output via a sigmoid.
        logits = self.classifier(x_flat)
        return jax.nn.sigmoid(logits)