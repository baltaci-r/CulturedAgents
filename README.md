# <img src="assets/emoji.png" alt="Custom Emoji" width="25" height="25"> Conformity, Confabulation, and Impersonation: Persona Inconstancy in Multi-Agent LLM Collaboration



```
@inproceedings{
    baltaji2024conformity,
    title={Conformity, Confabulation, and Impersonation: Persona Inconstancy in Multi-Agent {LLM} Collaboration},
    author={Razan Baltaji and Babak Hemmatian and Lav R. Varshney},
    booktitle={The 2nd Workshop on Cross-Cultural Considerations in NLP},
    year={2024},
    url={https://openreview.net/forum?id=2zH4HAq850}
}
```

Submitted to [arXiv](https://arxiv.org/abs/2405.03862) on May 6, 2024.

_Abstract:_ This study explores the sources of instability in maintaining cultural personas and opinions within multi-agent LLM systems. Drawing on simulations of inter-cultural collaboration and debate, we analyze agents' pre- and post-discussion private responses alongside chat transcripts to assess the stability of cultural personas and the impact of opinion diversity on group outcomes. Our findings suggest that multi-agent discussions can encourage collective decisions that reflect diverse perspectives, yet this benefit is tempered by the agents' susceptibility to conformity due to perceived peer pressure and challenges in maintaining consistent personas and opinions. Counterintuitively, instructions that encourage debate in support of one's opinions increase the rate of inconstancy. Without addressing the factors we identify, the full potential of multi-agent frameworks for producing more culturally diverse AI outputs will remain untapped. 


<div align="center">
  <img src="assets/teaser.png" alt="teaser" width="400"/>
</div>

## Usage

For installing all dependencies: 
```
pip install -e .
```

For generating multi agent debates for debate:

```
bash run.sh debate opinion
```

For generating multi agent debates for collaboration:
```
bash run.sh collab opinion
```

For compiling results for debate:
```
python compile.py debate opinion
```

For compiling results for collaboration:
```
python compile.py collab opinion
```