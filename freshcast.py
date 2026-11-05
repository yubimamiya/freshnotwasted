"""FreshCast: forecasting and stocking engine for fresh food (prototype on synthetic data).

build(storm=False) returns everything the website shows:
  - per store, per product: demand forecast with an 80% range, what to order, why
  - markdowns, donation pickups, staff hours, storm actions
  - a back-test: the tool vs. the current ordering rule on days the model never saw
"""
import math, datetime as dt
import numpy as np

TODAY = dt.date(2026, 11, 5)
START = dt.date(2024, 1, 1)

# name, dept, avg units/day, unit cost $, price $, shelf life (days), case size, kg/unit,
# holiday peaks {event: x normal}, storm pre-buy multiplier, linked social trend
P = [
 ("Bananas","Produce",120,.25,.59,6,40,.18,{},1.3,None),
 ("Strawberries 1lb","Produce",45,1.8,3.98,3,8,.45,{"valentines":2.0,"july4":1.4},1.0,"dubai_chocolate"),
 ("Bagged salad","Produce",40,1.6,3.48,5,12,.3,{"thanksgiving":1.3},0.9,None),
 ("Cucumbers","Produce",30,.35,.88,7,24,.3,{},1.0,"cucumber_salad"),
 ("Avocados","Produce",50,.6,1.18,4,48,.2,{"superbowl":3.0,"cinco":2.5},1.0,None),
 ("Fresh cranberries","Produce",3,1.5,2.98,21,24,.34,{"thanksgiving":9.0,"christmas":4.0},1.0,None),
 ("Sweet potatoes","Produce",25,.5,1.08,21,40,.3,{"thanksgiving":4.0,"christmas":2.0},1.1,None),
 ("White bread","Bakery",70,1.1,2.48,6,12,.6,{"july4":1.3},2.6,None),
 ("Sourdough loaf","Bakery",18,1.9,4.48,3,8,.7,{},1.3,None),
 ("Dinner rolls 12ct","Bakery",20,1.4,3.48,4,12,.5,{"thanksgiving":4.5,"christmas":3.0,"easter":2.5},1.2,None),
 ("Pumpkin pie","Bakery",6,3.0,6.98,4,6,.9,{"thanksgiving":10.0,"christmas":3.0},1.0,None),
 ("Chocolate cake","Bakery",8,6.0,13.98,5,4,1.0,{"valentines":2.5},1.0,"dubai_chocolate"),
 ("Chicken breast","Meat",55,5.5,9.48,5,6,1.1,{"july4":1.3},1.8,"protein"),
 ("Ground beef","Meat",50,4.5,6.98,4,8,.45,{"superbowl":1.6,"july4":1.8},1.8,None),
 ("Fresh whole turkey","Meat",2,14,24.0,7,4,6.0,{"thanksgiving":25.0,"christmas":6.0},1.0,None),
 ("Salmon fillet","Meat",15,7.0,11.98,3,6,.45,{"easter":1.5},1.0,None),
 ("Milk gallon","Dairy",110,2.3,3.48,12,4,3.9,{"thanksgiving":1.3},2.4,None),
 ("Eggs dozen","Dairy",85,2.0,3.28,28,15,.7,{"thanksgiving":1.5,"christmas":1.6,"easter":2.5},2.2,"protein"),
 ("Cottage cheese","Dairy",14,1.6,2.98,14,12,.45,{},1.2,"cottage_cheese"),
 ("Heavy cream","Dairy",12,2.0,3.98,14,12,.47,{"thanksgiving":3.5,"christmas":3.0},1.3,None),
 ("Charcuterie tray","Deli",8,6.5,12.98,7,6,.5,{"superbowl":3.0,"christmas":4.0,"newyear":3.5},1.0,"snack_plate"),
 ("Guacamole fresh","Deli",10,2.2,4.48,5,12,.4,{"superbowl":4.0,"cinco":3.0},1.0,None),
 ("Rotisserie chicken","Deli",45,3.5,4.98,1,8,1.0,{},1.6,None),
 ("Sushi rolls","Deli",12,3.5,7.98,1,8,.3,{},0.7,None),
]
ELAST = {"Produce":-1.8,"Bakery":-1.5,"Meat":-2.0,"Dairy":-1.2,"Deli":-1.6}
TRENDS = {  # topic: ([(start, days to peak, peak size 0-100)], demand effect at index 100)
 "dubai_chocolate": ([(dt.date(2025,1,5),20,90),(dt.date(2026,10,25),14,100)], 0.45),
 "cucumber_salad": ([(dt.date(2024,7,10),15,80)], 0.6),
 "cottage_cheese": ([(dt.date(2024,4,1),60,60),(dt.date(2025,9,1),40,50)], 0.5),
 "snack_plate": ([(dt.date(2025,5,1),30,70)], 0.35),
 "protein": ([(dt.date(2025,1,1),90,40)], 0.12),
}
TREND_LABEL = {"dubai_chocolate":"Dubai chocolate strawberry cups","cucumber_salad":"Cucumber salad recipe videos",
 "cottage_cheese":"Cottage cheese protein recipes","snack_plate":"Snack-plate dinners","protein":"High-protein meal prep"}
TREND_POSTS = {"dubai_chocolate":["Tried the Dubai chocolate strawberry cup everyone is posting 🍓🍫","Strawberries + pistachio cream + chocolate = my new obsession","Where are people finding good strawberries for this?"],
 "cucumber_salad":["Ate a whole cucumber again thanks to that salad video"],"cottage_cheese":["Cottage cheese flatbread actually works"],
 "snack_plate":["Snack plate dinner again, no regrets"],"protein":["Meal prep Sunday: 5 lbs of chicken breast"]}
PARTNERS = [("Northside Community Food Pantry",2.1,True),("St. Anne's Soup Kitchen",3.4,False),("Riverbend Food Bank",6.8,True)]
STORES = {"Store 101 · Supercenter":1.0,"Store 102 · Neighborhood Market":0.5,"Store 103 · Supercenter":1.25}
EVENT_NAMES = {"thanksgiving":"Thanksgiving","christmas":"Christmas","newyear":"New Year's Eve","easter":"Easter","july4":"July 4th",
 "superbowl":"Super Bowl","valentines":"Valentine's Day","cinco":"Cinco de Mayo"}
# minutes per task (labor standards; prototype assumptions)
MIN_PER_CASE_STOCK, MIN_PER_CASE_ROTATE, MIN_MARKDOWN_LOT, MIN_MARKDOWN_UNIT = 4.0, 0.4, 2.0, 0.15
MIN_DONATE_UNIT, MIN_DONATE_PICKUP, MIN_STORM_ITEM = 0.3, 10.0, 12.0

def _nth_wd(y,m,wd,n):
    d=dt.date(y,m,1); d+=dt.timedelta((wd-d.weekday())%7); return d+dt.timedelta(7*(n-1))
def _easter(y):
    a=y%19;b=y//100;c=y%100;d=(19*a+b-b//4-((b-(b+8)//25+1)//3)+15)%30;e=(32+2*(b%4)+2*(c//4)-d-c%4)%7;f=d+e-7*((a+11*d+22*e)//451)+114
    return dt.date(y,f//31,f%31+1)
EV = {}
for _y in range(2023,2028):
    for _k,_v in {"thanksgiving":_nth_wd(_y,11,3,4),"christmas":dt.date(_y,12,25),"newyear":dt.date(_y,12,31),"easter":_easter(_y),
       "july4":dt.date(_y,7,4),"superbowl":_nth_wd(_y,2,6,2),"valentines":dt.date(_y,2,14),"cinco":dt.date(_y,5,5)}.items():
        EV.setdefault(_k,[]).append(_v)
EVLIST = list(EV)
def days_until(d,ev):
    fut=[(x-d).days for x in EV[ev] if (x-d).days>=-2]; return min(fut) if fut else 99
def hol_weight(k):  # build-up to the event, peak 1-2 days before, dip right after
    if k>10: return 0.0
    if k<0: return -0.25
    if k==0: return 0.5
    return math.exp(-((k-1.5)**2)/(2*2.2**2))

DAYS=[START+dt.timedelta(i) for i in range((TODAY-START).days+15)]
N=len(DAYS); iT=(TODAY-START).days
DOY=np.array([d.timetuple().tm_yday for d in DAYS]); DOW=np.array([d.weekday() for d in DAYS])
DOWM=np.array([.88,.85,.9,.95,1.1,1.25,1.07])
EVK={e:np.array([days_until(d,e) for d in DAYS]) for e in EVLIST}
HW={e:np.array([hol_weight(k) for k in EVK[e]]) for e in EVLIST}
PAST_STORMS=[dt.date(2024,1,12),dt.date(2025,1,9),dt.date(2025,2,15),dt.date(2026,1,24)]

FEAT_NAMES=(["base"]+[f"dow{k}" for k in range(1,7)]+["season_s","season_c","weather","promo"]
  +[f"{e}:{w}" for e in EVLIST for w in ["after","day","peak","ramp","early"]]+[f"storm{o}" for o in range(-2,3)]+["trend","growth"])
def driver_group(n):
    if n.startswith("dow"): return "Day of week"
    if n.startswith("season"): return "Time of year"
    if n=="weather": return "Weather"
    if n=="promo": return "Promotion"
    if n.startswith("storm"): return "Storm warning"
    if n=="trend": return "Social media trend"
    if ":" in n: return EVENT_NAMES[n.split(":")[0]]
    return None

def _inv_norm(p):
    a=0.147; x=2*p-1; ln=math.log(1-x*x); t=2/(math.pi*a)+ln/2
    return math.sqrt(2)*math.copysign(math.sqrt(math.sqrt(t*t-ln/a)-t),x)


class World:
    """One synthetic world (normal or storm). Same seed so both scenarios share history."""
    def __init__(self, storm):
        self.storm=storm; rng=np.random.default_rng(7)
        self.temp=50-25*np.cos(2*np.pi*(DOY-15)/365)+np.convolve(rng.normal(0,6,N),np.ones(3)/3,"same")
        storms=PAST_STORMS+([TODAY+dt.timedelta(1)] if storm else [])
        self.phase=np.full(N,99)
        for s in storms:
            i=(s-START).days
            for o in range(-2,3):
                if 0<=i+o<N: self.phase[i+o]=o
        self.trucks=np.ones(N,bool); self.trucks[(self.phase==0)|(self.phase==1)]=False  # storms cancel 2 days of deliveries
        self.ti={}
        for k,(bursts,_) in TRENDS.items():
            x=np.zeros(N)
            for st,pk,sz in bursts:
                t=np.array([(d-st).days for d in DAYS],float)
                x+=np.where(t<0,0,np.where(t<pk,sz*t/pk,sz*np.exp(-(t-pk)/25)))
            self.ti[k]=np.clip(x+rng.normal(0,2,N),0,100)
        self.rng=rng

    def demand(self, p, scale):
        name,cat,base,cost,price,shelf,case,kg,hol,sm,tr=p; rng=self.rng
        mu=base*scale*DOWM[DOW]*(1+0.02*np.arange(N)/365)
        if name.startswith("Straw"): mu=mu*(1+0.35*np.exp(-((DOY-150)**2)/(2*30**2)))
        if cat=="Produce": mu=mu*(1+0.004*(self.temp-50))
        for e,L in hol.items(): mu=mu*np.maximum(0.7,1+(L-1)*HW[e])
        mu=mu*np.array([{-2:1+0.3*(sm-1),-1:sm,0:0.45,1:0.85,2:1.05}.get(int(ph),1.0) for ph in self.phase])
        if tr: mu=mu*(1+TRENDS[tr][1]*np.roll(self.ti[tr],3)/100)
        promo=(rng.random(N//7+1)<0.08).repeat(7)[:N]
        mu=mu*np.where(promo,0.8**ELAST[cat],1.0)
        return rng.poisson(rng.gamma(10,mu/10)).astype(float), promo

    def X(self, p, promo, idx):
        tr=p[10]
        cols=[np.ones(len(idx))]+[(DOW[idx]==k)*1.0 for k in range(1,7)]
        cols+=[np.sin(2*np.pi*DOY[idx]/365),np.cos(2*np.pi*DOY[idx]/365),(self.temp[idx]-50)/20,promo[idx]*1.0]
        for e in EVLIST:  # holiday markers: days-until-event windows
            k=EVK[e][idx]; cols+=[((k>=a)&(k<=b))*1.0 for a,b in [(-2,-1),(0,0),(1,2),(3,5),(6,10)]]
        for o in range(-2,3): cols.append((self.phase[idx]==o)*1.0)
        cols.append(np.roll(self.ti[tr],3)[idx]/100 if tr else np.zeros(len(idx)))
        cols.append(idx/365)
        return np.column_stack(cols)


def fit(X,y):
    """Per-product model: log-linear ridge regression; error range from residual quantiles."""
    yl=np.log1p(y); A=X.T@X+0.5*np.eye(X.shape[1]); A[0,0]-=0.5
    w=np.linalg.solve(A,X.T@yl); r=yl-X@w
    return w,np.quantile(r,[0.1,0.5,0.9])
def predict(w,rq,X):
    lp=X@w; return [np.expm1(lp+rq[k]).clip(0) for k in range(3)]

def service_level(p):
    name,cat,base,cost,price,shelf,case,kg,hol,sm,tr=p
    essential=sm>=1.8; cu=price-cost+(0.5*price if essential else 0); co=0.6*cost  # markdowns + donation recover ~40% of cost
    return min(0.97,max(0.75,cu/(cu+co)))


def simulate(p, y, q, world, a, b, smart, start_lots=None):
    """Run days a..b-1. smart=True: the tool (forecast ordering + markdown + donation). False: current rule."""
    name,cat,base,cost,price,shelf,case,kg,hol,sm,tr=p
    q10,q50,q90=q; sl=service_level(p); z=_inv_norm(sl)
    lots=[list(l) for l in start_lots] if start_lots else [[round(q50[a]*1.2),a+shelf-1]]
    m=dict(sold=0,lost=0,waste=0,donated=0,revenue=0.0,cost=0.0,md_units=0,stockout_days=0,demand=0)
    rng=np.random.default_rng(11); sales=list(y)
    for t in range(a,b):
        # demand: smart policy marks down units expiring today -> extra bargain demand for those only
        dem=int(y[t]); extra=0
        exp_today=sum(u for u,e in lots if e==t)
        if smart and exp_today>0: extra=int(rng.poisson(min(exp_today,0.3*q50[t])))
        sold=0; rev=0.0
        for need,discount_only in ((dem,False),(extra,True)):
            for lot in lots:
                if need<=0: break
                if discount_only and lot[1]!=t: continue
                take=min(lot[0],need); lot[0]-=take; need-=take; sold+=take
                rev+=take*price*(0.7 if (smart and lot[1]==t) else 1.0)
                if smart and lot[1]==t: m["md_units"]+=take
            if not discount_only:
                m["lost"]+=need; m["stockout_days"]+=need>0
        m["sold"]+=sold; m["revenue"]+=rev; m["demand"]+=dem; sales[t]=min(dem,sold)
        left=sum(u for u,e in lots if e==t)
        if smart: m["donated"]+=round(left*0.85); m["waste"]+=left-round(left*0.85)
        else: m["waste"]+=left
        lots=[l for l in lots if l[1]>t and l[0]>0]
        # order tonight for delivery tomorrow (if a truck is coming)
        if t+1<N and world.trucks[t+1]:
            usable=sum(u for u,e in lots)
            if smart:
                k=1
                while t+1+k<N and not world.trucks[t+1+k] and k<4: k+=1   # cover until the next truck
                mean=sum(q50[t+1+j] for j in range(k)); sd=math.sqrt(sum(((q90[t+1+j]-q10[t+1+j])/2.56)**2 for j in range(k)))
                target=mean+z*sd
                target=min(target,mean*1.0+ q50[t+1]*shelf)  # never more than can sell before spoiling
            else:
                target=1.1*sales[t-6]  # current rule: same day last week + 10%
            need=max(0,target-usable); cases=math.ceil(need/case-0.3) if need>0 else 0
            if cases: lots.append([cases*case,t+1+shelf-1]); m["cost"]+=cases*case*cost
    m["lots"]=lots; m["profit"]=m["revenue"]-m["cost"]
    return m


def build(storm=False):
    W=World(storm)
    out={"today":str(TODAY),"storm":storm,"stores":{},"trends":[],"events":[],"backtest":{}}
    bt_win=(iT-28,iT); storm_win=((dt.date(2026,1,24)-START).days-5,(dt.date(2026,1,24)-START).days+5)
    bt={k:{"tool":{},"current":{}} for k in ("recent","storm")}
    example_storm=None
    for sname,scale in STORES.items():
        rows=[]; actions=[]; donations=[]; markdowns=[]
        labor={"Receive & stock deliveries":0.0,"Rotate & check dates":0.0,"Markdowns":0.0,"Donation prep":0.0,"Storm prep":0.0}
        for p in P:
            name,cat,base,cost,price,shelf,case,kg,hol,sm,tr=p
            y,promo=W.demand(p,scale)
            # ---- back-tests: train only on days before each test window
            for key,(a,b) in (("recent",bt_win),("storm",storm_win)):
                w0,rq0=fit(W.X(p,promo,np.arange(a)),y[:a]); qa=[np.zeros(N)]*3
                pr=predict(w0,rq0,W.X(p,promo,np.arange(a,min(N,b+5))))
                qa=[np.concatenate([np.zeros(a),v,np.zeros(N-a-len(v))]) for v in pr]
                for pol in ("tool","current"):
                    m=simulate(p,y,qa,W,a,b,pol=="tool")
                    agg=bt[key][pol]
                    for f in ("sold","lost","waste","donated","revenue","cost","md_units","stockout_days","demand"): agg[f]=agg.get(f,0)+m[f]
                    agg["waste_kg"]=agg.get("waste_kg",0)+m["waste"]*kg; agg["waste_usd"]=agg.get("waste_usd",0)+m["waste"]*cost
                    agg["donated_kg"]=agg.get("donated_kg",0)+m["donated"]*kg
                    if key=="recent" and pol=="tool": end_lots=m["lots"]
                    if key=="storm" and name=="White bread" and sname.startswith("Store 101"):
                        example_storm=example_storm or {}
                        example_storm[pol]=m["lost"]
            # ---- live plan: model trained on all history up to yesterday
            w,rq=fit(W.X(p,promo,np.arange(iT)),y[:iT])
            fidx=np.arange(iT,iT+14); Xf=W.X(p,promo,fidx); q10,q50,q90=predict(w,rq,Xf)
            hb=np.arange(iT-28,iT); w0,rq0=fit(W.X(p,promo,np.arange(iT-28)),y[:iT-28])
            pb=predict(w0,rq0,W.X(p,promo,hb))[1]; err=float(np.mean(np.abs(pb-y[hb]))/max(1,np.mean(y[hb])))
            contrib={}
            for j,n in enumerate(FEAT_NAMES):
                g=driver_group(n)
                if g: contrib[g]=contrib.get(g,0)+w[j]*Xf[1,j]
            drivers=sorted([(g,round((math.exp(v)-1)*100)) for g,v in contrib.items() if abs(v)>0.04],key=lambda t:-abs(t[1]))[:4]
            # stock this morning = what the tool's simulated store had left last night + this morning's delivery
            lots=[[u,(e-iT)] for u,e in end_lots]  # expiry as days from today (0 = expires tonight)
            on_hand=sum(u for u,_ in lots)
            sl=service_level(p); z=_inv_norm(sl); essential=sm>=1.8
            cover=[1] if not storm else [0,1,2]
            if storm:
                usable=on_hand
            else:  # what's left for tomorrow after today's expected sales (oldest first)
                rem=q50[0]; left=[]
                for u,e in sorted(lots,key=lambda l:l[1]):
                    take=min(u,rem); rem-=take
                    if u-take>0 and e>=1: left.append(u-take)
                usable=sum(left)
            mean=sum(q50[c] for c in cover); sd=math.sqrt(sum(((q90[c]-q10[c])/2.56)**2 for c in cover))
            target=mean+z*sd; lo=max(0,mean-1.28*sd); hi=mean+1.28*sd
            need=max(0,target-usable); cases=math.ceil(need/case-0.3) if need>0 else 0; order=cases*case
            row=dict(product=name,dept=cat,on_hand=int(on_hand),usable=int(usable),fc=round(mean),lo=round(lo),hi=round(hi),
                order=order,cases=cases,case=case,target=round(target),sl=round(sl*100),err=round(err*100),drivers=drivers,shelf=shelf,
                stockout_risk=round(100*(1-_phi((usable+order-mean)/max(sd,1e-6)))),
                hist=[int(v) for v in y[iT-42:iT]],f50=[round(float(v),1) for v in q50],f10=[round(float(v),1) for v in q10],f90=[round(float(v),1) for v in q90],
                trend=TREND_LABEL.get(tr))
            rows.append(row)
            labor["Receive & stock deliveries"]+=cases*MIN_PER_CASE_STOCK; labor["Rotate & check dates"]+=on_hand/case*MIN_PER_CASE_ROTATE
            if storm:
                gap=target-usable
                if gap>0 and essential:
                    actions.append(dict(group="Storm",title=f"{name}: get {math.ceil(gap)} more units before the storm",
                        detail=f"No trucks for 2 days. Expected demand today through Saturday: {round(lo)}–{round(hi)}. On hand: {int(usable)}. "
                               f"Ask the distribution center for a same-day top-up and set a limit of 2 per customer.",qty=math.ceil(gap),mins=MIN_STORM_ITEM))
                    labor["Storm prep"]+=MIN_STORM_ITEM
            elif cases>0:
                actions.append(dict(group="Order",title=f"Order {cases} case{'s' if cases>1 else ''} of {name} ({order} units)",
                    detail=f"Tomorrow's demand: about {round(mean)}, likely between {round(lo)} and {round(hi)}. Usable on hand: {int(usable)}.",qty=order,mins=cases*MIN_PER_CASE_STOCK))
            # markdowns + donations: units expiring today/tomorrow that probably won't sell
            ahead=0
            for u,e in sorted(lots,key=lambda l:l[1]):
                if u<=0: continue
                if e<=1 and not (storm and essential):
                    exp_dem=sum(q50[:e+1])-ahead
                    unsold=max(0.0,u-max(0.0,exp_dem))
                    if unsold>=max(2,0.15*u):
                        disc=30 if unsold<0.5*u else 50
                        rec=round(min(u,exp_dem+unsold*0.6)*price*(1-disc/100))
                        markdowns.append(dict(product=name,units=int(u),expires="today" if e==0 else "tomorrow",unsold=round(unsold),disc=disc,recover=rec))
                        mins=MIN_MARKDOWN_LOT+MIN_MARKDOWN_UNIT*u; labor["Markdowns"]+=mins
                        actions.append(dict(group="Markdown",title=f"Mark down {int(u)} {name} by {disc}%",
                            detail=f"Expires {'tonight' if e==0 else 'tomorrow'}. About {round(unsold)} would not sell at full price.",qty=int(u),mins=round(mins,1)))
                        left=round(unsold*0.4)
                        if left>=2:
                            pt=next(x for x in PARTNERS if x[2] or cat in ("Produce","Bakery"))
                            donations.append(dict(product=name,units=left,kg=round(left*kg,1),partner=pt[0],miles=pt[1],
                                cold=pt[2],pickup=(TODAY+dt.timedelta(e+1)).strftime("%a %b %d, 9 AM"),notify="Today"))
                            labor["Donation prep"]+=MIN_DONATE_UNIT*left
                ahead+=u
        if donations: labor["Donation prep"]+=MIN_DONATE_PICKUP*len({d["partner"] for d in donations})
        for d in donations:
            actions.append(dict(group="Donate",title=f"Set aside {d['units']} {d['product']} ({d['kg']} kg) for {d['partner']}",
                detail=f"Pickup {d['pickup']}. Partner gets notice today so they can plan.",qty=d["units"],mins=round(MIN_DONATE_UNIT*d["units"],1)))
        if storm:
            actions.insert(0,dict(group="Storm",title="Stop markdowns on bread, milk, eggs and meat",detail="These will sell at full price during the storm rush.",qty=0,mins=5))
            actions.insert(1,dict(group="Storm",title="Release backroom stock in 3 rounds: now, 6 PM, tomorrow 7 AM",detail="Keeps shelves from going empty tonight while trucks are delayed.",qty=0,mins=30))
            actions.insert(2,dict(group="Storm",title="Bakery: bake 2 extra batches of white bread and rolls tonight",detail="In-store baking is the only supply that doesn't depend on trucks.",qty=0,mins=50))
            labor["Storm prep"]+=85
        hrs={k:round(v/60,1) for k,v in labor.items()}
        order_rank={"Storm":0,"Order":1,"Markdown":2,"Donate":3}
        actions.sort(key=lambda a:order_rank[a["group"]])
        out["stores"][sname]=dict(rows=rows,actions=actions,labor=hrs,donations=donations,markdowns=markdowns,
            kpi=dict(orders=sum(1 for r in rows if r["cases"]>0),cases=sum(r["cases"] for r in rows),markdowns=len(markdowns),
                     donate_kg=round(sum(d["kg"] for d in donations),1),hours=round(sum(hrs.values()),1)))
    for key in bt:
        for pol in bt[key]:
            v=bt[key][pol]; v["fill"]=round(100*v["sold"]/max(1,v["demand"]),1)
            v["profit"]=round(v["revenue"]-v["cost"]);
            for f in list(v):
                if isinstance(v[f],float): v[f]=round(v[f],1)
    out["backtest"]=dict(recent=bt["recent"],storm=bt["storm"],recent_label=f"{DAYS[bt_win[0]]:%b %d} – {DAYS[bt_win[1]-1]:%b %d, %Y}",
                         storm_label="Jan 19 – 28, 2026 (snowstorm Jan 24)",bread=example_storm)
    for k in TRENDS:
        out["trends"].append(dict(key=k,topic=TREND_LABEL[k],now=round(W.ti[k][iT-1]),wk=round(W.ti[k][iT-1]-W.ti[k][iT-8]),
            products=[p[0] for p in P if p[10]==k],series=[round(v) for v in W.ti[k][iT-60:iT]],posts=TREND_POSTS[k],effect=round(TRENDS[k][1]*W.ti[k][iT-1])))
    for e in EVLIST:
        k=days_until(TODAY,e)
        if 0<=k<=90: out["events"].append(dict(event=EVENT_NAMES[e],days=k,date=f"{TODAY+dt.timedelta(k):%a %b %d}",
            products=[f"{p[0]} (×{p[8][e]:g})" for p in P if e in p[8] and p[8][e]>=1.5]))
    out["events"].sort(key=lambda e:e["days"])
    return out

def _phi(x): return 0.5*(1+math.erf(x/math.sqrt(2)))

if __name__=="__main__":
    import json,sys
    d=build("--storm" in sys.argv)
    print(json.dumps({k:d["backtest"][k] for k in ("recent","storm")},indent=1))
    s=d["stores"]["Store 101 · Supercenter"]; print(s["kpi"]); [print(" -",a["title"]) for a in s["actions"][:8]]
