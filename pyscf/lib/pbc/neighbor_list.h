#if !defined(HAVE_DEFINED_NEIGHBORPAIR_H)
#define HAVE_DEFINED_NEIGHBORPAIR_H
typedef struct NeighborPair_struct {
    int nimgs;
    int *Ls_list;
} NeighborPair;
#endif

#if !defined(HAVE_DEFINED_NEIGHBORLIST_H)
#define HAVE_DEFINED_NEIGHBORLIST_H
typedef struct NeighborList_struct {
    int nish;
    int njsh;
    int nimgs;
    NeighborPair **pairs;
} NeighborList;
#endif

