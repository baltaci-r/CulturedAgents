import os, sys, math, json
from collections import Counter, defaultdict

import pandas as pd, numpy as np 

import seaborn as sns, plotly.express as px, matplotlib.pyplot as plt

from scipy.stats import entropy
from plotly.subplots import make_subplots
from plotly.offline import plot
import plotly.graph_objects as go

from utils import * 
from builders import * 


class Discussion: 

    def __init__(self, id):
        self.id=id
        self.discussion=[]
        self.agents=[]
        self.regions=[]
        self.onboarding_opinions=defaultdict(str)
        self.reflection_opinions=defaultdict(str)
        self.intermediate_opinions=defaultdict(list)
        self.intermediate_replies=defaultdict(list)
        self.intermediate_opinions_with_order=[]
        self.gratitude=defaultdict(int)
        self.terminate=defaultdict(int)
        self.analyzing=defaultdict(int)
        self.first_opinion=None
        self.group_opinion=None
        self.num_agents=0

    def get_freq(self, x):
        counter = Counter(self.onboarding_opinions.values())
        return counter[x]/sum(counter.values()) 
    
    def get_num_replies(self):
        return sum([len(v)for v in self.intermediate_opinions.values()])
    
    @staticmethod
    def _get_entropy(values):
        counter = Counter(values)
        h = 0 
        for v in counter.values():
            p = v/sum(counter.values())
            h += -p*math.log(p, 2)
        return h

    def get_entropy(self, step):
        if step == 'onboarding_opinion':
            values = self.onboarding_opinions.values()
        elif step == 'reflection_opinion':
            values = self.reflection_opinions.values()
        elif step =='intermediate_opinion':
            values = [v for vs in self.intermediate_opinions.values() for v in vs]
        elif step=='region': 
            values = self.regions
        else: 
            raise NotImplementedError
        
        return self._get_entropy(values)

    def ratio_reflection_diff_from_onboarding(self):
        return sum([1 if (self.reflection_opinions[c]!=ob and self.reflection_opinions[c] is not None) else 0 for c, ob in self.onboarding_opinions.items()])/self.num_agents
    
    def ratio_intermediate_diff_from_onboarding(self):
        res = []
        for c, opinions in self.intermediate_opinions.items():
            res.append(sum([1 if self.onboarding_opinions[c]!=a else 0 for a in opinions])/len(opinions)) 
        return sum(res)/self.num_agents
    
    def onboarding_compared_to_reflection(self):
        res = []
        for c, ob in self.onboarding_opinions.items():
            rf = self.reflection_opinions[c]
            if ob==None or rf== None:
                change_onboarding_reflection='N/A'
            elif rf!=ob:
                change_onboarding_reflection=1
            else:
                change_onboarding_reflection=0

            if ob==None or self.intermediate_opinions[c]==[]:
                ratio_intermediate_equal_onboarding='N/A'
                ratio_intermediate_equal_reflection='N/A'
            else:
                ratio_intermediate_equal_onboarding=self.intermediate_opinions[c].count(ob)/len(self.intermediate_opinions[c])
                ratio_intermediate_equal_reflection=self.intermediate_opinions[c].count(rf)/len(self.intermediate_opinions[c])
           
            res.append((change_onboarding_reflection, self.get_freq(ob), ratio_intermediate_equal_onboarding, ratio_intermediate_equal_reflection))
        return res
    
    def group_opinion_compared_to_reflection(self):
        res = []
        for c, ob in self.onboarding_opinions.items():
            rf = self.reflection_opinions[c]
            if ob==None or rf== None:
                change_onboarding_reflection='N/A'
            elif rf!=ob:
                change_onboarding_reflection=1
            else:
                change_onboarding_reflection=0

            if self.group_opinion==None or self.intermediate_opinions[c]==[]:
                ratio_intermediate_equal_group_opinion='N/A'
                ref_equal_group_opinion = 'N/A'
            else:
                ratio_intermediate_equal_group_opinion= self.intermediate_opinions[c].count(self.group_opinion)/len(self.intermediate_opinions[c])
                if self.group_opinion == rf:
                    ref_equal_group_opinion = 1
                else:
                    ref_equal_group_opinion = 0 

            res.append((change_onboarding_reflection, self.get_freq(ob), ref_equal_group_opinion, ratio_intermediate_equal_group_opinion))
        return res

    def group_opinion_same_as_initiator(self): 
        return self.group_opinion == list(self.discussion[0].values())[0]
    
    def initiator_changes_opinion(self): 
        initiator, discussion_opinion = list(self.discussion[0].items())[0]
        opinion = self.onboarding_opinions[initiator]
        return True if discussion_opinion != opinion else False

    def initiator_opinion_prob(self): 
        initiator, discussion_opinion = list(self.discussion[0].items())[0]
        opinion = self.onboarding_opinions[initiator]
        return self.get_freq(opinion), self.get_freq(discussion_opinion)
    
    def detect_impersonation(self): 
        false_agent_count = 0 
        for name, replies in self.intermediate_replies.items():
            name = name.replace('_', ' ')[:-6]
            for reply in replies:
                matches = re.findall("As an? ([\w\s]+) agent", reply)
                for match in matches:
                    if match != name:
                        print(self.id)
                        false_agent_count+=1
                        print(name, '->', match)
                        pass
                    # print(f'[{self.intermediate_replies[ind-1][0]}]', self.intermediate_replies[ind-1][1])
                    # print('-'*80)
                    # print(f'[{self.intermediate_replies[ind][0]}]', self.intermediate_replies[ind][1])
                    # print('*'*80)

        return false_agent_count
                    
    def detect_confabulation(self):
        unseen_opinions = 0 
        intermediate_opinions = [a for ans in self.intermediate_opinions.values() for a in ans]
        previous_ans = set(intermediate_opinions +  list(self.onboarding_opinions.values()))
        for ans in  self.reflection_opinions.values():
            if ans and ans not in previous_ans: # and ans in self.chars:
                print(ans, previous_ans)
                unseen_opinions+=1
        return unseen_opinions
    
def get_ratio_over_bin(df, x, col, total='size'):
    return x[total] / df[df[col]==x[col]][total].sum()


mode = sys.argv[1]
filter = sys.argv[2]
scale = 3

assert mode in ['collab', 'debate']

save_dir = f"runs/{mode}/{filter}/" 
results_dir = f"results/{mode}/{filter}/" 
os.makedirs(results_dir, exist_ok=True)
results_dir = results_dir + f"{mode}-{filter}-"

exceptions = 0
no_group_opinion = 0 
no_reflection_opinion = 0 
no_onboarding_opinion = 0 

discussions = []

examples = os.listdir(os.path.join(save_dir))
for e in sorted(examples):
    try: 
        j = json.load(open(os.path.join(save_dir, e)))
    except Exception as ex:
        print(ex)
        pass
    
    example = Example(j['example'])
    group_opinion = j['group_opinion']['opinion']
    onboarding = j['onboarding']
    reflection = j['reflection']
    agents = j['onboarding'].keys()

    if j['exception'] : 
        exceptions += 1
        print(os.path.join(save_dir, e), len(agents))
        os.remove(os.path.join(save_dir, e))

    if not group_opinion:
        no_group_opinion +=1
        continue
    else:
        discussion = Discussion(e[:-5])
        discussion.group_opinion = group_opinion
        discussion.agents = agents
        discussion.num_agents = len(agents)
        if len(agents) != 5:
            print(os.path.join(save_dir, e), len(agents))
            os.remove(os.path.join(save_dir, e))
            
        discussion.onboarding_opinions = onboarding

        d = j['discussion']
           
        for ind, m in enumerate(d):
            name = m["name"]
            opinion = m["opinion"]
            content = m["content"]
            discussion.discussion.append({name: opinion})
            if opinion:
                discussion.intermediate_opinions[name].append(opinion)
                discussion.intermediate_opinions_with_order.append((name, opinion))
                discussion.intermediate_replies[name].append(content)

        discussion.reflection_opinions=reflection        
        discussions.append(discussion)


print('Exceptions', exceptions)
print('Number of examples', len(discussions))


filter = 'onboarding_opinion' if filter == 'opinion' else 'region'
onboarding_entropy = np.array([d.get_entropy(filter) for d in discussions]) # also in examples
intermediate_ents = np.array([d.get_entropy('intermediate_opinion') for d in discussions])
reflection_ents = np.array([d.get_entropy('reflection_opinion') for d in discussions])
region_ents = np.array([d.get_entropy('region') for d in discussions])


num_replies = [d.get_num_replies() for d in discussions]
group_opinion_freq = np.array([d.get_freq(d.group_opinion) for d in discussions])
group_opinion_same_as_initiator = [d.group_opinion_same_as_initiator() for d in discussions]
initiator_changes_opinion = [d.initiator_changes_opinion() for d in discussions] 
initiator_opinion_prob = [d.initiator_opinion_prob() for d in discussions]

ratio_reflection_diff_from_onboarding = np.array([d.ratio_reflection_diff_from_onboarding() for d in discussions])
ratio_intermediate_diff_from_onboarding = np.array([d.ratio_intermediate_diff_from_onboarding() for d in discussions])
onboarding_compared_to_reflection = [d.onboarding_compared_to_reflection() for d in discussions]
group_opinion_compared_to_reflection = [d.group_opinion_compared_to_reflection() for d in discussions]

false_agent = [d.detect_impersonation() for d in discussions]
unseen_answers = [d.detect_confabulation() for d in discussions]

false_agent_ratio = [i/n for i, n in zip(false_agent, num_replies) if n]
print("False agent percentage:", round(sum(false_agent_ratio)/len(false_agent_ratio)*100,2), '%')
print("Unseen answers percentage:", round(sum(unseen_answers)/(5*len(unseen_answers))*100,2), '%')

##### Figure 4:  Initiator Changes Opinion

df = pd.DataFrame({
    'onboarding_entropy': onboarding_entropy, 'initiator_changes_opinion': initiator_changes_opinion, 
    })
df['onboarding_entropy'] = df['onboarding_entropy'].round(2).map(str)
df = df.groupby(['onboarding_entropy', 'initiator_changes_opinion'], as_index=False, dropna=False).size()
df['ratio'] = df.apply(lambda x: get_ratio_over_bin(df, x, 'onboarding_entropy'), axis=1)
df = df.sort_values(['onboarding_entropy', 'initiator_changes_opinion'], ascending=[True, False])
fig = px.bar(df, x="onboarding_entropy", y="ratio", color = 'initiator_changes_opinion', labels={
                     "onboarding_entropy": "Entropy",
                     "initiator_changes_opinion": "",
                     "ratio": "Ratio"
                 },
            )
labels = {'True': r'$I \neq O$', 'False': '$I = O$'}
fig.for_each_trace(lambda t: t.update(name = labels[t.name]))
fig.update_layout(
    font_family="Times New Roman",
    title_font_family="Times New Roman",
    font_size=16
)
fig.update_layout(
        margin={'t':40,'l':40,'b':40,'r':10}
    )

fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="center",
    x=0.5
))

fig.write_image(f'{results_dir}initiator_changes_opinion.png', scale=scale)
fig.write_html(f'{results_dir}initiator_changes_opinion.html')
fig.show()


##### Figure 3: Initiators Dominate Group Prediction

df = pd.DataFrame({'onboarding_entropy': onboarding_entropy, 'group_opinion_same_as_initiator': group_opinion_same_as_initiator})
df['onboarding_entropy'] = df['onboarding_entropy'].round(2).map(str)
df = df.groupby(['onboarding_entropy', 'group_opinion_same_as_initiator'], as_index=False, dropna=False).size()
df['ratio'] = df.apply(lambda x: get_ratio_over_bin(df, x, 'onboarding_entropy'), axis=1)
df = df[df['size']!=0]
df = df.sort_values(['onboarding_entropy', 'group_opinion_same_as_initiator', 'ratio'], ascending=[True, False, True])
fig = px.bar(df, x="onboarding_entropy", y="ratio", color="group_opinion_same_as_initiator", 
            labels={
                     "onboarding_entropy": "Entropy",
                     "group_opinion_same_as_initiator": "",
                     "ratio":  "Ratio"
                 },
            )
labels = {'True': r'$G = I$', 'False': r'$G \neq I$'}
fig.for_each_trace(lambda t: t.update(name = labels[t.name]))
fig.update_layout(
    font_family="Times New Roman",
    title_font_family="Times New Roman",
    font_size=16
)
fig.update_layout(
        margin={'t':40,'l':40,'b':40,'r':10}
    )
fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="center",
    x=0.5
))
fig.write_image(f'{results_dir}group_opinion_same_as_initiator.png', scale=scale)
fig.write_html(f'{results_dir}group_opinion_same_as_initiator.html')
fig.show()


##### Figure 2: Group Opinion follows the Distribution of Opinions during onboarding 

df = pd.DataFrame({'onboarding_entropy': onboarding_entropy, 'group_opinion_freq': group_opinion_freq})
df['onboarding_entropy'] = df['onboarding_entropy'].round(2)
df['onboarding_entropy'] = df['onboarding_entropy'].map(str)
df = df.groupby(['onboarding_entropy', 'group_opinion_freq'], as_index=False, dropna=False).size()
df['ratio'] = df.apply(lambda x: get_ratio_over_bin(df, x, 'onboarding_entropy'), axis=1)
df = df[df['size']!=0]
df = df.sort_values(['onboarding_entropy', 'group_opinion_freq', 'ratio'])
fig = px.bar(df, x="onboarding_entropy", y="ratio", color="group_opinion_freq", 
             color_continuous_scale="bluered", 
             labels={
                     "onboarding_entropy": "Entropy",
                     "ratio": "Ratio", 
                     "group_opinion_freq": "Probability"
                 },
            )
fig.update_layout(
    font_family="Times New Roman",
    title_font_family="Times New Roman",
    font_size=14
)
fig.update_layout(
        margin={'t':40,'l':40,'b':40,'r':10}
    )
fig.write_image(f'{results_dir}onboarding_probability_by_entropy.png', scale=scale)
fig.write_html(f'{results_dir}onboarding_probability_by_entropy.html')
fig.show()


##### Tables:

df = pd.DataFrame({'onboarding_entropy': onboarding_entropy, 'onboarding_compared_to_reflection': onboarding_compared_to_reflection})
df['onboarding_entropy'] = df['onboarding_entropy'].round(2)
df = df.explode('onboarding_compared_to_reflection')
df[['change_onboarding_reflection', 'prob_onboarding','ratio_intermediate_equal_onboarding', 'ratio_intermediate_equal_reflection']] = pd.DataFrame(df.onboarding_compared_to_reflection.tolist(), index= df.index)
df.reset_index(inplace=True)
df = df.drop(columns=['onboarding_compared_to_reflection', 'index'])
df['Count'] = 1
df = df.groupby(['onboarding_entropy', 'prob_onboarding', 'change_onboarding_reflection', 'ratio_intermediate_equal_onboarding', 'ratio_intermediate_equal_reflection']).Count.count().reset_index()
df = df[df['change_onboarding_reflection']!='N/A']
df = df[df['ratio_intermediate_equal_onboarding']!='N/A']


print('Peer Pressure', mode)

for ind, ((entropy, prob_onboarding), group) in enumerate(df.groupby(['onboarding_entropy', 'prob_onboarding', ])):
    for change_onboarding_reflection, _group in group.groupby(['change_onboarding_reflection']):
        sum_ratio_intermediate_equal_reflection = group['ratio_intermediate_equal_reflection'].map(float) * _group['Count']
        average_ratio_intermediate_nequal_reflection = sum_ratio_intermediate_equal_reflection.sum()/_group['Count'].sum()
        ratio = _group['Count'].sum()/group['Count'].sum()
        if not change_onboarding_reflection[0]:
            print(",".join(map(str, [entropy, prob_onboarding, round(100*ratio,2), round(average_ratio_intermediate_nequal_reflection,2)])), end='')
        else:
            print(','+",".join(map(str, [round(100*ratio,2), round(average_ratio_intermediate_nequal_reflection,2)])))


df = pd.DataFrame({'onboarding_entropy': onboarding_entropy, 'group_opinion_compared_to_reflection': group_opinion_compared_to_reflection})
df['onboarding_entropy'] = df['onboarding_entropy'].round(2)
df = df.explode('group_opinion_compared_to_reflection')
df[['change_onboarding_reflection', 'prob_onboarding','ref_equal_group_opinion', 'ratio_intermediate_equal_group_opinion']] = pd.DataFrame(df.group_opinion_compared_to_reflection.tolist(), index= df.index)
df.reset_index(inplace=True)
df = df.drop(columns=['group_opinion_compared_to_reflection', 'index'])
df['Count'] = 1
df = df.groupby(['onboarding_entropy', 'prob_onboarding', 'change_onboarding_reflection', 'ref_equal_group_opinion', 'ratio_intermediate_equal_group_opinion']).Count.count().reset_index()
df = df[df['change_onboarding_reflection']!='N/A']
df = df[df['ratio_intermediate_equal_group_opinion']!='N/A']

print('Peer Influence', mode)

for ind, ((entropy, prob_onboarding), group) in enumerate(df.groupby(['onboarding_entropy', 'prob_onboarding', ])):
    for change_onboarding_reflection, group1 in group.groupby(['change_onboarding_reflection']):
        for ref_equal_group_opinion, group2 in group1.groupby(['ref_equal_group_opinion']):
            sum_ratio_intermediate_equal_group_opinion = group2['ratio_intermediate_equal_group_opinion'].map(float) * group2['Count']
            average_ratio_intermediate_equal_group_opinion = sum_ratio_intermediate_equal_group_opinion.sum()/group1['Count'].sum()
            ratio1 = group1['Count'].sum()/group['Count'].sum()
            ratio2 = group2['Count'].sum()/group1['Count'].sum()
            if not change_onboarding_reflection[0]:
                if not ref_equal_group_opinion[0]:
                    print(",".join(map(str, [entropy, prob_onboarding, round(100*ratio1, 2), round(100*ratio2,2)])), end=',')
                else:
                    print(f'{round(100*ratio2,2)}', end=',')
            else:
                if not ref_equal_group_opinion[0]:
                    print(",".join(map(str, [round(100*ratio1, 2), round(100*ratio2,2)])), end=',')
                else:
                    print(f'{round(100*ratio2,2)}', end='\n')
