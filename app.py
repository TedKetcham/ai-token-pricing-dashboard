import os
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from research import collect_documents,load_documents,answer_question,propose_updates,approve_updates,HISTORY,DATA

st.set_page_config(page_title='AI Token Pricing Intelligence',page_icon='💠',layout='wide')
st.title('💠 AI Token Pricing Intelligence')
st.caption('Decision support for businesses selecting text-generation APIs | USD per 1 million tokens')

@st.cache_data(ttl=60)
def read_prices():return pd.read_csv(DATA)
df=read_prices()
try: secret_key=st.secrets.get('OPENAI_API_KEY','')
except Exception:secret_key=''
key=os.environ.get('OPENAI_API_KEY') or secret_key
with st.sidebar:
    st.header('Scenario & filters')
    providers=st.multiselect('Providers',sorted(df.provider.unique()),default=sorted(df.provider.unique()))
    requests=st.number_input('Requests per month',min_value=1,value=100000,step=1000)
    input_tokens=st.number_input('Input tokens per request',min_value=0,value=1000,step=100)
    output_tokens=st.number_input('Output tokens per request',min_value=0,value=500,step=100)
    st.caption('Estimated cost = requests × (input tokens × input rate + output tokens × output rate) / 1,000,000')
    if not key:
        key=st.text_input('OpenAI API key (optional, session only)',type='password',help='Never commit your key to GitHub. For deployed apps use Streamlit Secrets.')
    st.caption('Standard paid text API rates only. Excludes caching, batch, tool fees, taxes, and exceptional context tiers.')
filtered=df[df.provider.isin(providers)].copy()
if filtered.empty:st.warning('Select at least one provider.');st.stop()
filtered['monthly_cost']=requests*(input_tokens*filtered.input_per_m+output_tokens*filtered.output_per_m)/1_000_000
winner=filtered.loc[filtered.monthly_cost.idxmin()]
last_verification=str(df.verified_date.max())
st.info(f'**Approved pricing snapshot:** {last_verification}. Prices are stored locally; official documents are fetched on demand. Fetching a page is not the same as verifying its prices. See Source verification tab.')
a,b,c,d=st.columns(4)
a.metric('Models compared',len(filtered));b.metric('Cheapest for this workload',winner['model']);c.metric('Lowest monthly cost',f'${winner.monthly_cost:,.2f}');d.metric('Potential savings vs priciest',f'${filtered.monthly_cost.max()-winner.monthly_cost:,.2f}')
t1,t2,t3,t4,t5=st.tabs(['📊 Compare','🧮 Business case','🤖 RAG research assistant','🔄 Source verification','📚 Sources & history'])
with t1:
    metric=st.radio('Compare by',['Monthly cost','Input $/1M','Output $/1M'],horizontal=True)
    col={'Monthly cost':'monthly_cost','Input $/1M':'input_per_m','Output $/1M':'output_per_m'}[metric]
    fig=px.bar(filtered.sort_values(col),x=col,y='model',color='provider',orientation='h',text_auto='.2f',title=metric,hover_data=['input_per_m','output_per_m'])
    fig.update_layout(height=max(410,65*len(filtered)),yaxis_title='',xaxis_title='USD')
    st.plotly_chart(fig,width="stretch")
    st.dataframe(filtered[['provider','model','input_per_m','output_per_m','monthly_cost','source_url']],hide_index=True,width="stretch",column_config={'source_url':st.column_config.LinkColumn('Official source'),'monthly_cost':st.column_config.NumberColumn(format='$%.2f')})
    st.caption('Cost-only ranking; quality, speed, safety, and model availability must be evaluated separately.')
with t2:
    st.subheader('Procurement scenario')
    st.write(f'**{requests:,} monthly requests** × **{input_tokens:,} input** + **{output_tokens:,} output tokens** per request.')
    baseline=st.selectbox('Compare against',filtered.model.tolist())
    base=float(filtered.loc[filtered.model==baseline,'monthly_cost'].iloc[0]);s=filtered[['provider','model','monthly_cost']].copy();s['savings_vs_baseline']=base-s.monthly_cost
    st.dataframe(s.sort_values('monthly_cost'),hide_index=True,width="stretch",column_config={'monthly_cost':st.column_config.NumberColumn(format='$%.2f'),'savings_vs_baseline':st.column_config.NumberColumn(format='$%.2f')})
    st.markdown(f'**Decision insight:** Under this workload, **{winner["model"]}** is the lowest listed token-cost option at **${winner.monthly_cost:,.2f}/month**. This is a budgeting comparison, not a quality recommendation.')
    st.warning('A real vendor decision also requires testing response quality, latency, compliance, and availability. Token costs alone cannot determine best overall value.')
with t3:
    st.subheader('Retrieval-augmented research assistant')
    docs=load_documents();ok=[d for d in docs if d['status']=='ok']
    st.caption(f'{len(ok)} official provider documents available locally. Refresh them in the next tab before asking questions. Answers use retrieved excerpts and cite source URLs.')
    question=st.text_input('Ask a pricing question',value='How do the providers differ in their published text token prices? What are the limitations of this comparison?')
    if st.button('Answer using retrieved official sources',type='primary'):
        if not key:
            st.error('Enter an OpenAI API key in the sidebar, or configure OPENAI_API_KEY in Streamlit Secrets.')
        elif not ok:
            st.error('Fetch official sources under Source verification first.')
        else:
            try:
                with st.spinner('Retrieving source excerpts and asking the AI...'):
                    answer,evidence=answer_question(question,docs,filtered,key)
                st.session_state['rag_result']={'answer':answer,'evidence':evidence,'question':question}
            except Exception as exc:
                st.session_state.pop('rag_result',None)
                st.error(f'RAG request failed: {type(exc).__name__}: {exc}')
    result=st.session_state.get('rag_result')
    if result:
        st.markdown('### AI research answer')
        st.markdown(result['answer'])
        with st.expander('Show retrieved evidence and timestamps'):
            for e in result['evidence']:
                st.markdown(f'**{e["provider"]}** — [{e["url"]}]({e["url"]}) — fetched {e["fetched_at_utc"]}')
                st.caption(e['text'][:950]+'…')
    st.caption('RAG pipeline: official webpage retrieval → text chunking → keyword relevance ranking → grounded OpenAI response → linked evidence. The LLM does not browse the web independently.')
with t4:
    st.subheader('Official source collection and verification')
    st.write('Step 1 fetches public pricing documentation. Step 2 asks an LLM to propose numeric updates. **A human must review and approve proposals** before the approved price table changes.')
    if st.button('1 · Fetch official pricing pages'):
        with st.spinner('Fetching three official sources...'):st.session_state['docs']=collect_documents()
    docs=st.session_state.get('docs',load_documents())
    for doc in docs:
        status='✅' if doc['status']=='ok' else '⚠️'
        st.write(f'{status} **{doc["provider"]}** · {doc["fetched_at_utc"]} · {doc["status"]} · [Official page]({doc["url"]})')
    if st.button('2 · Extract candidate price updates',disabled=not bool(key)):
        with st.spinner('Extracting documented standard text rates...'):
            st.session_state['proposals'],st.session_state['errors']=propose_updates(key,docs,df)
    proposals=st.session_state.get('proposals',[])
    for error in st.session_state.get('errors',[]):st.warning(error)
    if proposals:
        st.dataframe(pd.DataFrame(proposals),hide_index=True,width="stretch")
        st.warning('LLM extraction can be wrong. Verify every proposed rate against the linked official source before approval. Missing/ambiguous models are not changed.')
        approved=st.multiselect('Select individually verified rows to approve',options=[p['model'] for p in proposals])
        confirm=st.checkbox('I manually checked these prices against the official provider pages.')
        if st.button('3 · Approve selected verified updates',disabled=not(confirm and approved)):
            n=approve_updates([p for p in proposals if p['model'] in approved]);read_prices.clear();st.success(f'{n} approved price rows saved. Reload the app to see updated metrics.');st.session_state['proposals']=[]
    elif not key:st.caption('An API key is required for LLM-assisted price extraction. Official-page fetching works without one.')
    st.caption('On Streamlit Community Cloud, filesystem changes may be temporary and are not shared across replicas. For persistent scheduled updates, use the included GitHub Actions workflow and repository secrets.')
with t5:
    st.subheader('Source register')
    st.dataframe(df[['provider','model','verified_date','source_url','notes']],hide_index=True,width="stretch",column_config={'source_url':st.column_config.LinkColumn('Official documentation')})
    st.write('**Scope:** Google, OpenAI and Anthropic publish directly comparable general-purpose text API token rates. NVIDIA and Apple remain important ecosystem companies but are outside this narrowly defined first-party text-token API comparison.')
    st.write('**Research limitations:** Standard prices are not always comparable for longer contexts, reasoning tokens, multimodal inputs, caching, discounts, or promotional rates. This dataset is a dated approved snapshot, not a live billing quote.')
    if HISTORY.exists():
        history=pd.read_csv(HISTORY)
        st.dataframe(history.tail(50),hide_index=True,width="stretch")
        if len(history):
            h=history.copy();h['snapshot_utc']=pd.to_datetime(h['snapshot_utc'],errors='coerce')
            st.plotly_chart(px.line(h.dropna(subset=['snapshot_utc']),x='snapshot_utc',y='input_per_m',color='model',markers=True,title='Recorded input-price snapshots (only actual recorded observations)'),width="stretch")
    else:st.caption('No historical observations recorded yet. The first approved update creates a baseline snapshot; no historical trends are fabricated.')
    st.download_button('Download approved pricing CSV',df.to_csv(index=False),file_name='prices.csv',mime='text/csv')
