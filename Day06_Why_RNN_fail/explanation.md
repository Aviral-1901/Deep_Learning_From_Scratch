## Why RNN fail for long sequence
RNN works by keeping a fixed vector state that compresses every information it has obtained so far. The state is of fixed size so as the sequence gets longer all the information had to be squeezed in the small space so either the infromation from earlier timesteps gets erased or the information remains so little that it is difficult to get the full idea. Also RNN has time steps for each word so gradients flowing back have to pass through many time steps shrinking with each one causing vanishing gradient problem.                                                         
RNN carries everything forward in limited memory.

## Core idea of transformer
Words can directly look upto each other which preserves the information making it remain intact unlike the compressed or lost information of RNN. Becaue every word accesses every other word directly, nothing gets compressed and gradients do not have to travel through time steps to reach early words.