## Why RNN fail for long sequence
RNN works by keeping a fixed vector state that compresses every information it has obtained so far. The state is of fixed size so as the sequence gets longer all the information had to be squeezed in the small space so either the infromation from earlier timesteps gets erased or the information remains so little that it is difficult to get the full idea. Also RNN has time steps for each word so gradients flowing back have to pass through many time steps shrinking with each one causing vanishing gradient problem.                                                         
RNN carries everything forward in limited memory.                                                         
RNN is sequential so it cannnot be processed in parallel so even if we are able to compute large data in parallel we are stuck because of the RNN working.

## Core idea of transformer
Words can directly look upto each other which preserves the information making it remain intact unlike the compressed or lost information of RNN. Becaue every word accesses every other word directly, nothing gets compressed and gradients do not have to travel through time steps to reach early words.              
In transformers words have direct access to other words so the the problem of long sequences is solved. Because all words are processed simultaneously rather than sequentially, so transformers can use GPU parallelism well. So they scale so more data and more hardware both help.


## 3 matrix of transformers 
- Library Analogy
- Query(Q) : the question we ask when walking into the library with what are we looking for (what a word is looking for when trying to find its reference)
- Key(K) : The title of every book, what each book tells about itself (what the word tells about itself)
- Value(V) : The actual content inside the book (the real meaning the word contributes when it is attended)