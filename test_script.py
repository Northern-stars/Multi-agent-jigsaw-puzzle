import copy
def cross_permute(self,permutation_a,last_outsider_a,permutation_b,last_outsider_b):
        new_permutation_a=copy.deepcopy(permutation_a)
        new_permutation_b=copy.deepcopy(permutation_b)
        a_swap=(permutation_a[-1],last_outsider_a)
        b_swap=(last_outsider_b,permutation_b[-1])
        a_swap_flag=(a_swap[0]!=a_swap[1]) and (-1 not in a_swap)
        b_swap_flag=(b_swap[0]!=b_swap[1]) and (-1 not in b_swap)
        # print(f"a_swap_flag: {a_swap_flag}, b_swap_flag: {b_swap_flag}")
        if -1 in new_permutation_a[0:9]:
            invalid_index=new_permutation_a.index(-1)
            new_permutation_a[invalid_index]=new_permutation_a[-1]
            new_permutation_a[-1]=-1


        if -1 in new_permutation_b[0:9]:
            invalid_index=new_permutation_b.index(-1)
            new_permutation_b[invalid_index]=new_permutation_b[-1]
            new_permutation_b[-1]=-1

        if (not a_swap_flag) and (not b_swap_flag):
                return new_permutation_a,new_permutation_b
        if a_swap==b_swap and a_swap_flag and b_swap_flag:
                last_outsider_a_index=permutation_b.index(last_outsider_a)
                last_outsider_b_index=permutation_a.index(last_outsider_b)
                new_permutation_a[last_outsider_b_index]=permutation_b[-1]
                new_permutation_b[last_outsider_a_index]=permutation_a[-1]
                new_permutation_a[-1]=-1
                new_permutation_b[-1]=-1
                return new_permutation_a,new_permutation_b
        if a_swap_flag :
                last_outsider_a_index=new_permutation_b.index(a_swap[1])
                last_outsider_b_index=new_permutation_a.index(a_swap[0])
                new_permutation_a[last_outsider_b_index]=permutation_b[-1]
                new_permutation_b[last_outsider_a_index]=permutation_a[-1]
        if b_swap_flag:
                last_outsider_a_index=new_permutation_b.index(b_swap[1])
                last_outsider_b_index=new_permutation_a.index(b_swap[0])
                new_permutation_a[last_outsider_b_index]=permutation_b[-1]
                new_permutation_b[last_outsider_a_index]=permutation_a[-1]
        return new_permutation_a,new_permutation_b

print(cross_permute(self=1,
              permutation_a=[0,1,2,9,4,5,6,7,8,3],
              last_outsider_a=9,
              permutation_b=[3,10,11,12,13,14,15,16,17,9],
              last_outsider_b=3))