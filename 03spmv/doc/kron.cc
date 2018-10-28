
typedef struct blocked_sparse {
  blocked_sparse_node_kind_t kind;
  union {
    sparse_t block;
    sparse_t * children[2];
  };
} blocked_sparse_t;

sparse_t coo_to_blocked(sparse_t A) {
  idx_t nnz = A.nnz;
  if (nnz < max_leaf_nnz) {
    blocked_sparse_t b = { blocked_sparse_node_kind_leaf, block = A };
  } else {
    idx_t Mh = A.M / 2;
    idx_t Nh = A.N / 2;
    for (idx_t k = 0; k < nnz; k++) {
      coo_elem_t * e = A.elems + k;
      idx_t i = e->i;
      idx_t j = e->j;
      c[i>=Mh][j>=Nh]++;
    }
  }
  
  for (idx_t k = 0; k < nnz; k++) {
    
  }
  
  assert(A.format == sparse_format_coo ||
         A.format == sparse_format_coo_sorted);
  idx_t nnz = A.nnz;
  coo_elem_t * B_elems = 0;
  if (in_place) {
    B_elems = A.coo.elems;
  } else {
    B_elems = (coo_elem_t *)xalloc(sizeof(coo_elem_t) * nnz);
    memcpy(B_elems, A.coo.elems, sizeof(coo_elem_t) * nnz);
  }
  if (A.format == sparse_format_coo) {
    qsort((void*)B_elems, nnz, sizeof(coo_elem_t), coo_elem_cmp);
  }
  coo_t coo = { B_elems };
  sparse_t B = { sparse_format_coo_sorted, A.M, A.N, A.nnz, { .coo = coo } };
  return B;
}


kron(A, B) {
  sparse_iterator_t iA = iter_start(A);
  for (coo_elem_t eA = sparse_iterator_next(iA); eA.i >= 0; eA = sparse_iterator_next(iA)) {
    sparse_iterator_t iB = iter_start(A);
    for (coo_elem_t eB = sparse_iterator_next(iB); eB.i >= 0; eB = sparse_iterator_next(iB)) {
      i = i0 * M1 + i1;
      j = j0 * N1 + j1;
    }
  }
}
