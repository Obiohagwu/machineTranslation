# machineTranslation
Machine translation task with transformer architecture

### On Attention
The crux of the transformer architecture is based on this mechanism of attention; yes, attention. That thing that allows you focus on a particular piece of information in order to achive a task, whether that just entails paying attention during a conversatoin, or studying the ins and out of a new exciting topic or field. *Attention is all you need!*

The self-attention mechanism can be defined as:

![Screen Shot 2022-05-22 at 12 50 53 PM](https://user-images.githubusercontent.com/73560826/169706500-44586f80-cf47-4fd4-a330-ff0e2705dfc8.png)

It entails taking in as input, 3 variables V - values[i]; K - key[i] and Q - query. Thing of it as analagous to a hash table where a key[i] maps to a value[i] given a query. The main caveat is that a euclidean similarity function is usually applies to the query and the key[i] then multiplied by the value[i] to attain the correct output i.e the attention "score".
