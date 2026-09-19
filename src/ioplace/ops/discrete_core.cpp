// Sparse categorical conditional expectation. Geometry construction is separate.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include <stdexcept>

struct NetMetric {
  int32_t io=0, ft=0;
  double hpwl=0.0, tree_wl=0.0;
  uint64_t passed=0;
};

struct Lattice {
  int size;
  const int16_t* ids;
  double xl,yl,xh,yh;
  int index(double v,double low,double high) const {
    double value=(v-low)/((high-low)/size);
    if(value<=0) return 0;
    if(value>=size-1) return size-1;
    if(!std::isfinite(value)) throw std::overflow_error("nonfinite lattice index");
    return int(value);
  }
  int ix(double x) const { return index(x,xl,xh); }
  int iy(double y) const { return index(y,yl,yh); }
  int at(int x,int y) const { return ids[size_t(y)*size+x]; }
  int rid(double x,double y) const { return at(ix(x),iy(y)); }
  int line(int x0,int y0,int x1,int y1,uint64_t& mask) const {
    int count=0,previous=-1;
    if(y0==y1) {
      for(int x=std::min(x0,x1);x<=std::max(x0,x1);++x) {
        int r=at(x,y0);mask|=uint64_t(1)<<r;
        if(previous>=0 && previous!=r) ++count;
        previous=r;
      }
    } else {
      for(int y=std::min(y0,y1);y<=std::max(y0,y1);++y) {
        int r=at(x0,y);mask|=uint64_t(1)<<r;
        if(previous>=0 && previous!=r) ++count;
        previous=r;
      }
    }
    return count;
  }
};

static NetMetric measure_net(int e,const int32_t* starts,const int32_t* nodes,
    const double* ox,const double* oy,const double* x,const double* y,const Lattice& grid) {
  NetMetric metric;
  int begin=starts[e],degree=starts[e+1]-begin;
  if(degree<=1) return metric;
  std::vector<double> px(degree),py(degree),best(degree,std::numeric_limits<double>::infinity());
  std::vector<int> parent(degree,-1);
  std::vector<uint8_t> used(degree,0);
  uint64_t terminal=0;
  for(int p=0;p<degree;++p) {
    px[p]=x[nodes[begin+p]]+ox[begin+p];py[p]=y[nodes[begin+p]]+oy[begin+p];
    if(!std::isfinite(px[p]) || !std::isfinite(py[p])) throw std::overflow_error("nonfinite pin position");
    terminal|=uint64_t(1)<<grid.rid(px[p],py[p]);
  }
  metric.hpwl=*std::max_element(px.begin(),px.end())-*std::min_element(px.begin(),px.end())
             +*std::max_element(py.begin(),py.end())-*std::min_element(py.begin(),py.end());
  if(!std::isfinite(metric.hpwl)) throw std::overflow_error("nonfinite net span");
  best[0]=0;
  for(int step=0;step<degree;++step) {
    int next=-1;double cost=std::numeric_limits<double>::infinity();
    for(int p=0;p<degree;++p) if(!used[p] && best[p]<cost) { next=p;cost=best[p]; }
    if(next<0) throw std::overflow_error("nonfinite MST distance");
    used[next]=1;
    if(parent[next]>=0) {
      int a=parent[next],b=next;
      metric.tree_wl+=std::abs(px[a]-px[b])+std::abs(py[a]-py[b]);
      metric.io+=grid.line(grid.ix(px[a]),grid.iy(py[a]),grid.ix(px[b]),grid.iy(py[a]),metric.passed);
      metric.io+=grid.line(grid.ix(px[b]),grid.iy(py[a]),grid.ix(px[b]),grid.iy(py[b]),metric.passed);
    }
    for(int p=0;p<degree;++p) if(!used[p]) {
      double distance=std::abs(px[next]-px[p])+std::abs(py[next]-py[p]);
      if(distance<best[p]) { best[p]=distance;parent[p]=next; }
    }
  }
  metric.ft=__builtin_popcountll(metric.passed&~terminal);
  return metric;
}

extern "C" int ioplace_net_metrics(int factors,int lattice_size,
    const int32_t* starts,const int32_t* nodes,const double* ox,const double* oy,
    const double* x,const double* y,const int16_t* region_grid,
    double xl,double yl,double xh,double yh,
    int32_t* io,int32_t* ft,double* hpwl,double* tree_wl,uint64_t* passed) {
  try {
    Lattice grid{lattice_size,region_grid,xl,yl,xh,yh};
    for(int e=0;e<factors;++e) {
      if(starts[e+1]-starts[e]>256) return 2;
      auto metric=measure_net(e,starts,nodes,ox,oy,x,y,grid);
      io[e]=metric.io;ft[e]=metric.ft;hpwl[e]=metric.hpwl;
      tree_wl[e]=metric.tree_wl;passed[e]=metric.passed;
    }
    return 0;
  } catch(...) { return 1; }
}

extern "C" int ioplace_refine(int cells,int candidates,int factors,int regions,int lattice_size,int swaps,
    const int32_t* starts,const int32_t* nodes,const double* ox,const double* oy,
    double* x,double* y,const int16_t* region_grid,const int32_t* active,
    const double* candidate_x,const double* candidate_y,const int32_t* cell_start,
    const int32_t* factor_id,const int32_t* swap_left,const int32_t* swap_right,
    const double* area,const double* region_area,const double* target,const double* starting_load,
    const double* original_x,const double* original_y,
    double alpha,double beta,double balance_weight,double displacement_weight,
    double xl,double yl,double xh,double yh,
    int32_t* accepted_moves,int32_t* accepted_swaps,double* trace,double* final_load) {
  try {
    const int K=regions,C=candidates;
    Lattice grid{lattice_size,region_grid,xl,yl,xh,yh};
    std::vector<uint8_t> adjacent(size_t(K)*K,0);
    for(int yy=0;yy<lattice_size;++yy) for(int xx=0;xx<lattice_size;++xx) {
      int a=grid.at(xx,yy);
      if(xx+1<lattice_size) { int b=grid.at(xx+1,yy);adjacent[size_t(a)*K+b]=adjacent[size_t(b)*K+a]=1; }
      if(yy+1<lattice_size) { int b=grid.at(xx,yy+1);adjacent[size_t(a)*K+b]=adjacent[size_t(b)*K+a]=1; }
    }
    std::vector<NetMetric> metrics(factors);
    std::vector<double> load(starting_load,starting_load+K);
    double value=0.0;
    for(int e=0;e<factors;++e) {
      if(starts[e+1]-starts[e]>256) return 2;
      metrics[e]=measure_net(e,starts,nodes,ox,oy,x,y,grid);
      value+=alpha*metrics[e].io+beta*metrics[e].ft;
    }
    for(int k=0;k<K;++k)
      value+=balance_weight*(load[k]-target[k])*(load[k]-target[k])/(region_area[k]*region_area[k]);
    auto displacement=[&](int cell,double xx,double yy) {
      return std::abs(xx-original_x[cell])+std::abs(yy-original_y[cell]);
    };
    for(int i=0;i<cells;++i) value+=displacement_weight*displacement(i,x[active[i]],y[active[i]]);
    trace[0]=value;
    std::vector<int32_t> affected;
    std::vector<double> load_delta(K);
    auto collect=[&](int i,int j) {
      affected.assign(factor_id+cell_start[i],factor_id+cell_start[i+1]);
      if(j>=0) affected.insert(affected.end(),factor_id+cell_start[j],factor_id+cell_start[j+1]);
      std::sort(affected.begin(),affected.end());
      affected.erase(std::unique(affected.begin(),affected.end()),affected.end());
    };
    auto trial=[&](int i,double nx,double ny,int j,double jx,double jy,bool commit) {
      int node=active[i],other=j>=0 ? active[j] : node;
      double old_x=x[node],old_y=y[node],old_jx=x[other],old_jy=y[other];
      std::fill(load_delta.begin(),load_delta.end(),0.0);
      load_delta[grid.rid(old_x,old_y)]-=area[i];load_delta[grid.rid(nx,ny)]+=area[i];
      double delta=displacement_weight*(displacement(i,nx,ny)-displacement(i,old_x,old_y));
      if(j>=0) {
        load_delta[grid.rid(old_jx,old_jy)]-=area[j];load_delta[grid.rid(jx,jy)]+=area[j];
        delta+=displacement_weight*(displacement(j,jx,jy)-displacement(j,old_jx,old_jy));
      }
      for(int k=0;k<K;++k) delta+=balance_weight*(2*(load[k]-target[k])*load_delta[k]
          +load_delta[k]*load_delta[k])/(region_area[k]*region_area[k]);
      x[node]=nx;y[node]=ny;
      if(j>=0) { x[other]=jx;y[other]=jy; }
      for(int e:affected) {
        auto fresh=measure_net(e,starts,nodes,ox,oy,x,y,grid);
        delta+=alpha*(fresh.io-metrics[e].io)+beta*(fresh.ft-metrics[e].ft);
        if(commit) metrics[e]=fresh;
      }
      if(commit) {
        for(int k=0;k<K;++k) load[k]+=load_delta[k];
      } else {
        x[node]=old_x;y[node]=old_y;
        if(j>=0) { x[other]=old_jx;y[other]=old_jy; }
      }
      return delta;
    };
    for(int i=0;i<cells;++i) {
      collect(i,-1);
      int best=-1;double gain=-1e-12;
      for(int c=0;c<C;++c) {
        int current=grid.rid(x[active[i]],y[active[i]]);
        int next=grid.rid(candidate_x[size_t(i)*C+c],candidate_y[size_t(i)*C+c]);
        if(current!=next && !adjacent[size_t(current)*K+next]) continue;
        double delta=trial(i,candidate_x[size_t(i)*C+c],candidate_y[size_t(i)*C+c],-1,0,0,false);
        if(delta<gain) { gain=delta;best=c; }
      }
      accepted_moves[i]=best;
      if(best>=0) value+=trial(i,candidate_x[size_t(i)*C+best],candidate_y[size_t(i)*C+best],-1,0,0,true);
      trace[i+1]=value;
    }
    for(int s=0;s<swaps;++s) {
      int i=swap_left[s],j=swap_right[s];collect(i,j);
      int a=grid.rid(x[active[i]],y[active[i]]),b=grid.rid(x[active[j]],y[active[j]]);
      if(a==b || !adjacent[size_t(a)*K+b]) {
        accepted_swaps[s]=0;trace[cells+s+1]=value;continue;
      }
      double nx=x[active[j]],ny=y[active[j]],jx=x[active[i]],jy=y[active[i]];
      double delta=trial(i,nx,ny,j,jx,jy,false);
      accepted_swaps[s]=delta<-1e-12;
      if(accepted_swaps[s]) value+=trial(i,nx,ny,j,jx,jy,true);
      trace[cells+s+1]=value;
    }
    std::copy(load.begin(),load.end(),final_load);
    return 0;
  } catch(...) { return 1; }
}

extern "C" int ioplace_ce(
    int cells, int candidates, int regions, int factors, int incidences,
    const double* probabilities, const int32_t* candidate_region,
    const double* area, const double* region_area, const double* target,
    const double* inactive_load, const double* unary,
    const int32_t* cell_start, const int32_t* factor_id,
    const uint64_t* candidate_mask, const uint64_t* fixed_mask,
    const uint64_t* passed_mask, double alpha, double beta, double balance_weight,
    int32_t* choices, double* trace, double* identity_error) {
  try {
    const int K=regions, C=candidates;
    std::vector<double> logs(size_t(factors)*K,0.0), absent(size_t(incidences)*K);
    std::vector<int32_t> zeros(size_t(factors)*K,0);
    std::vector<double> mean(inactive_load,inactive_load+K), variance(K,0.0);
    std::vector<double> cell_region(size_t(cells)*K,0.0);
    double unary_total=0.0;
    for(int e=0;e<factors;++e) for(int k=0;k<K;++k)
      zeros[size_t(e)*K+k]=(fixed_mask[e]>>k)&1;
    for(int i=0;i<cells;++i) {
      for(int c=0;c<C;++c) {
        const double p=probabilities[size_t(i)*C+c];
        cell_region[size_t(i)*K+candidate_region[size_t(i)*C+c]]+=p;
        unary_total+=p*unary[size_t(i)*C+c];
      }
      for(int k=0;k<K;++k) {
        const double p=cell_region[size_t(i)*K+k];
        mean[k]+=area[i]*p;
        variance[k]+=area[i]*area[i]*p*(1-p);
      }
      for(int j=cell_start[i];j<cell_start[i+1];++j) for(int k=0;k<K;++k) {
        double q=0.0;
        for(int c=0;c<C;++c) if(!((candidate_mask[size_t(j)*C+c]>>k)&1))
          q+=probabilities[size_t(i)*C+c];
        absent[size_t(j)*K+k]=q;
        const size_t slot=size_t(factor_id[j])*K+k;
        if(q==0.0) ++zeros[slot]; else logs[slot]+=std::log(q);
      }
    }
    auto product=[&](size_t slot) { return zeros[slot] ? 0.0 : std::exp(logs[slot]); };
    auto coefficient=[&](int e,int k) { return beta*double((passed_mask[e]>>k)&1)-alpha; };
    double value=alpha*factors*(K-1)+unary_total;
    for(int e=0;e<factors;++e) for(int k=0;k<K;++k)
      value+=coefficient(e,k)*product(size_t(e)*K+k);
    for(int k=0;k<K;++k)
      value+=balance_weight*((mean[k]-target[k])*(mean[k]-target[k])+variance[k])/(region_area[k]*region_area[k]);
    trace[0]=value;
    *identity_error=0.0;
    std::vector<double> deltas(C), others, old_products, balance_delta(C);
    for(int i=0;i<cells;++i) {
      const int first=cell_start[i],last=cell_start[i+1];
      others.resize(size_t(last-first)*K);old_products.resize(others.size());
      double old_unary=0.0;
      for(int c=0;c<C;++c) old_unary+=probabilities[size_t(i)*C+c]*unary[size_t(i)*C+c];
      for(int j=first;j<last;++j) for(int k=0;k<K;++k) {
        const size_t slot=size_t(factor_id[j])*K+k,local=size_t(j-first)*K+k;
        old_products[local]=product(slot);
        const double q=absent[size_t(j)*K+k];
        const int other_zero=zeros[slot]-(q==0.0);
        others[local]=other_zero ? 0.0 : std::exp(logs[slot]-(q==0.0 ? 0.0 : std::log(q)));
      }
      for(int c=0;c<C;++c) {
        double delta=unary[size_t(i)*C+c]-old_unary;
        for(int j=first;j<last;++j) for(int k=0;k<K;++k) {
          const size_t local=size_t(j-first)*K+k;
          const bool present=(candidate_mask[size_t(j)*C+c]>>k)&1;
          delta+=coefficient(factor_id[j],k)*((present ? 0.0 : others[local])-old_products[local]);
        }
        double db=0.0;
        for(int k=0;k<K;++k) {
          const double p=cell_region[size_t(i)*K+k];
          const double step=area[i]*(double(candidate_region[size_t(i)*C+c]==k)-p);
          db+=(2*(mean[k]-target[k])*step+step*step-area[i]*area[i]*p*(1-p))/(region_area[k]*region_area[k]);
        }
        deltas[c]=delta+balance_weight*db;
      }
      double weighted=0.0;
      for(int c=0;c<C;++c) weighted+=probabilities[size_t(i)*C+c]*deltas[c];
      *identity_error=std::max(*identity_error,std::abs(weighted));
      const int chosen=int(std::min_element(deltas.begin(),deltas.end())-deltas.begin());
      choices[i]=chosen;
      for(int j=first;j<last;++j) for(int k=0;k<K;++k) {
        const size_t slot=size_t(factor_id[j])*K+k;
        const double q=absent[size_t(j)*K+k];
        if(q==0.0) --zeros[slot]; else logs[slot]-=std::log(q);
        if((candidate_mask[size_t(j)*C+chosen]>>k)&1) ++zeros[slot];
      }
      for(int k=0;k<K;++k) {
        const double p=cell_region[size_t(i)*K+k];
        mean[k]+=area[i]*(double(candidate_region[size_t(i)*C+chosen]==k)-p);
        variance[k]-=area[i]*area[i]*p*(1-p);
      }
      value+=deltas[chosen];trace[i+1]=value;
    }
    return 0;
  } catch(...) { return 1; }
}
