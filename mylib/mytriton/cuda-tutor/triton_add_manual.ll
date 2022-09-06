define void @add(float addrspace(1)* align 16 %aptr, float addrspace(1)* align 16 %bptr, float addrspace(1)* align 16 %cptr, i32 %N) {
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %tid_m128 = urem i32 %tid, 128
  %tid_m128_x4 = mul i32 %tid_m128, 4

  %idx_0 = add i32 %tid_m128_x4, 0
  %idx_1 = add i32 %tid_m128_x4, 1
  %idx_2 = add i32 %tid_m128_x4, 2
  %idx_3 = add i32 %tid_m128_x4, 3
  %idx_4 = add i32 %tid_m128_x4, 512
  %idx_5 = add i32 %tid_m128_x4, 513
  %idx_6 = add i32 %tid_m128_x4, 514
  %idx_7 = add i32 %tid_m128_x4, 515

  %blkid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %blkid_x1024 = mul i32 %blkid, 1024

  %abs_idx_0 = add i32 %idx_0, %blkid_x1024
  %abs_idx_1 = add i32 %idx_1, %blkid_x1024
  %abs_idx_2 = add i32 %idx_2, %blkid_x1024
  %abs_idx_3 = add i32 %idx_3, %blkid_x1024
  %abs_idx_4 = add i32 %idx_4, %blkid_x1024
  %abs_idx_5 = add i32 %idx_5, %blkid_x1024
  %abs_idx_6 = add i32 %idx_6, %blkid_x1024
  %abs_idx_7 = add i32 %idx_7, %blkid_x1024

  %cmp_res_0 = icmp slt i32 %abs_idx_0, %N
  %cmp_res_4 = icmp slt i32 %abs_idx_4, %N

  %aptr_0 = getelementptr float, float addrspace(1)* %aptr, i32 %abs_idx_0
  %aptr_4 = getelementptr float, float addrspace(1)* %aptr, i32 %abs_idx_4

  %a_left = call {i32, i32, i32, i32} asm sideeffect "@$4 ld.global.v4.b32 {$0, $1, $2, $3}, [$5];", "=r,=r,=r,=r,b,l"(i1 %cmp_res_0, float addrspace(1)* %aptr_0)
  %a0_i32 = extractvalue {i32, i32, i32, i32} %a_left, 0
  %a1_i32 = extractvalue {i32, i32, i32, i32} %a_left, 1
  %a2_i32 = extractvalue {i32, i32, i32, i32} %a_left, 2
  %a3_i32 = extractvalue {i32, i32, i32, i32} %a_left, 3
  %a0 = bitcast i32 %a0_i32 to float
  %a1 = bitcast i32 %a1_i32 to float
  %a2 = bitcast i32 %a2_i32 to float
  %a3 = bitcast i32 %a3_i32 to float

  %a_right = call {i32, i32, i32, i32} asm sideeffect "@$4 ld.global.v4.b32 {$0, $1, $2, $3}, [$5];", "=r,=r,=r,=r,b,l"(i1 %cmp_res_4, float addrspace(1)* %aptr_4)
  %a4_i32 = extractvalue {i32, i32, i32, i32} %a_right, 0
  %a5_i32 = extractvalue {i32, i32, i32, i32} %a_right, 1
  %a6_i32 = extractvalue {i32, i32, i32, i32} %a_right, 2
  %a7_i32 = extractvalue {i32, i32, i32, i32} %a_right, 3
  %a4 = bitcast i32 %a4_i32 to float
  %a5 = bitcast i32 %a5_i32 to float
  %a6 = bitcast i32 %a6_i32 to float
  %a7 = bitcast i32 %a7_i32 to float

  %bptr_0 = getelementptr float, float addrspace(1)* %bptr, i32 %abs_idx_0
  %bptr_4 = getelementptr float, float addrspace(1)* %bptr, i32 %abs_idx_4

  %b_left = call {i32, i32, i32, i32} asm sideeffect "@$4 ld.global.v4.b32 {$0, $1, $2, $3}, [$5];", "=r,=r,=r,=r,b,l"(i1 %cmp_res_0, float addrspace(1)* %bptr_0)
  %b0_i32 = extractvalue {i32, i32, i32, i32} %b_left, 0
  %b1_i32 = extractvalue {i32, i32, i32, i32} %b_left, 1
  %b2_i32 = extractvalue {i32, i32, i32, i32} %b_left, 2
  %b3_i32 = extractvalue {i32, i32, i32, i32} %b_left, 3
  %b0 = bitcast i32 %b0_i32 to float
  %b1 = bitcast i32 %b1_i32 to float
  %b2 = bitcast i32 %b2_i32 to float
  %b3 = bitcast i32 %b3_i32 to float

  %b_right = call {i32, i32, i32, i32} asm sideeffect "@$4 ld.global.v4.b32 {$0, $1, $2, $3}, [$5];", "=r,=r,=r,=r,b,l"(i1 %cmp_res_4, float addrspace(1)* %bptr_4)
  %b4_i32 = extractvalue {i32, i32, i32, i32} %b_right, 0
  %b5_i32 = extractvalue {i32, i32, i32, i32} %b_right, 1
  %b6_i32 = extractvalue {i32, i32, i32, i32} %b_right, 2
  %b7_i32 = extractvalue {i32, i32, i32, i32} %b_right, 3
  %b4 = bitcast i32 %b4_i32 to float
  %b5 = bitcast i32 %b5_i32 to float
  %b6 = bitcast i32 %b6_i32 to float
  %b7 = bitcast i32 %b7_i32 to float

  %c0 = fadd float %a0, %b0
  %c1 = fadd float %a1, %b1
  %c2 = fadd float %a2, %b2
  %c3 = fadd float %a3, %b3
  %c4 = fadd float %a4, %b4
  %c5 = fadd float %a5, %b5
  %c6 = fadd float %a6, %b6
  %c7 = fadd float %a7, %b7

  %cptr_0 = getelementptr float, float addrspace(1)* %cptr, i32 %abs_idx_0
  %cptr_4 = getelementptr float, float addrspace(1)* %cptr, i32 %abs_idx_4

  %c0_i32 = bitcast float %c0 to i32
  %c1_i32 = bitcast float %c1 to i32
  %c2_i32 = bitcast float %c2 to i32
  %c3_i32 = bitcast float %c3 to i32
  call void asm sideeffect "@$0 st.global.v4.b32 [$1], {$2, $3, $4, $5};", "b,l,r,r,r,r"(i1 %cmp_res_0, float addrspace(1)* %cptr_0, i32 %c0_i32, i32 %c1_i32, i32 %c2_i32, i32 %c3_i32)

  %c4_i32 = bitcast float %c4 to i32
  %c5_i32 = bitcast float %c5 to i32
  %c6_i32 = bitcast float %c6 to i32
  %c7_i32 = bitcast float %c7 to i32
  call void asm sideeffect "@$0 st.global.v4.b32 [$1], {$2, $3, $4, $5};", "b,l,r,r,r,r"(i1 %cmp_res_4, float addrspace(1)* %cptr_4, i32 %c4_i32, i32 %c5_i32, i32 %c6_i32, i32 %c7_i32)

  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()

!0 = !{void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32)* @add, !"kernel", i32 1}
!1 = !{void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32)* @add, !"maxntidx", i32 128}
!nvvm.annotations = !{!0, !1}
