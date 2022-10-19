# machineTranslation
Machine translation task with transformer architecture

### On Attention
The crux of the transformer architecture is based on this mechanism of attention; yes, attention. That thing that allows you focus on a particular piece of information in order to achive a task, whether that just entails paying attention during a conversatoin, or studying the ins and out of a new exciting topic or field. *Attention is all you need!*

The self-attention mechanism can be defined as:

![CodeCogsEqn (21)](https://user-images.githubusercontent.com/73560826/196820101-b7299aec-c5af-4414-a453-f86533675b29.svg)



It entails taking in as input, 3 variables V - values[i]; K - key[i] and Q - query. Thing of it as analagous to a hash table where a key[i] maps to a value[i] given a query. The main caveat is that a euclidean similarity function is usually applies to the query and the key[i] then multiplied by the value[i] to attain the correct output i.e the attention "score".
