; ModuleID = 'add_kernel__Pfp32_Pfp32_Pfp32_i32__4c1024'
source_filename = "add_kernel__Pfp32_Pfp32_Pfp32_i32__4c1024"

define void @add_kernel__Pfp32_Pfp32_Pfp32_i32__4c1024(float addrspace(1)* align 16 %0, float addrspace(1)* align 16 %1, float addrspace(1)* align 16 %2, i32 %3) {
entry:
  %4 = call i64 asm "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;", "=l"()
  %5 = call i64 asm "createpolicy.fractional.L2::evict_last.b64 $0, 1.0;", "=l"()
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %7 = urem i32 %6, 128
  %8 = mul i32 %7, 4
  %idx_0_0 = add i32 %8, 0
  %idx_0_1 = add i32 %8, 1
  %idx_0_2 = add i32 %8, 2
  %idx_0_3 = add i32 %8, 3
  %idx_0_4 = add i32 %8, 512
  %idx_0_5 = add i32 %8, 513
  %idx_0_6 = add i32 %8, 514
  %idx_0_7 = add i32 %8, 515
  %9 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %10 = mul i32 %9, 1024
  %11 = add i32 0, %8
  %12 = add i32 %11, 0
  %13 = add i32 0, %8
  %14 = add i32 %13, 1
  %15 = add i32 0, %8
  %16 = add i32 %15, 2
  %17 = add i32 0, %8
  %18 = add i32 %17, 3
  %19 = add i32 0, %8
  %20 = add i32 %19, 512
  %21 = add i32 0, %8
  %22 = add i32 %21, 513
  %23 = add i32 0, %8
  %24 = add i32 %23, 514
  %25 = add i32 0, %8
  %26 = add i32 %25, 515
  %27 = add i32 %10, %11
  %28 = add i32 %27, 0
  %29 = add i32 %10, %13
  %30 = add i32 %29, 1
  %31 = add i32 %10, %15
  %32 = add i32 %31, 2
  %33 = add i32 %10, %17
  %34 = add i32 %33, 3
  %35 = add i32 %10, %19
  %36 = add i32 %35, 512
  %37 = add i32 %10, %21
  %38 = add i32 %37, 513
  %39 = add i32 %10, %23
  %40 = add i32 %39, 514
  %41 = add i32 %10, %25
  %42 = add i32 %41, 515
  %43 = icmp slt i32 %28, %3
  %44 = icmp slt i32 %30, %3
  %45 = icmp slt i32 %32, %3
  %46 = icmp slt i32 %34, %3
  %47 = icmp slt i32 %36, %3
  %48 = icmp slt i32 %38, %3
  %49 = icmp slt i32 %40, %3
  %50 = icmp slt i32 %42, %3
  %51 = getelementptr float, float addrspace(1)* %0, i32 %27
  %52 = getelementptr float, float addrspace(1)* %51, i32 0
  %53 = getelementptr float, float addrspace(1)* %0, i32 %29
  %54 = getelementptr float, float addrspace(1)* %53, i32 1
  %55 = getelementptr float, float addrspace(1)* %0, i32 %31
  %56 = getelementptr float, float addrspace(1)* %55, i32 2
  %57 = getelementptr float, float addrspace(1)* %0, i32 %33
  %58 = getelementptr float, float addrspace(1)* %57, i32 3
  %59 = getelementptr float, float addrspace(1)* %0, i32 %35
  %60 = getelementptr float, float addrspace(1)* %59, i32 512
  %61 = getelementptr float, float addrspace(1)* %0, i32 %37
  %62 = getelementptr float, float addrspace(1)* %61, i32 513
  %63 = getelementptr float, float addrspace(1)* %0, i32 %39
  %64 = getelementptr float, float addrspace(1)* %63, i32 514
  %65 = getelementptr float, float addrspace(1)* %0, i32 %41
  %66 = getelementptr float, float addrspace(1)* %65, i32 515
  %67 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %68 = urem i32 %67, 32
  %69 = call { i32, i32, i32, i32 } asm sideeffect "@$4 ld.global.v4.b32 {$0,$1,$2,$3}, [ $5 + 0];", "=r,=r,=r,=r,b,l"(i1 %43, float addrspace(1)* %51)
  %70 = extractvalue { i32, i32, i32, i32 } %69, 0
  %71 = bitcast i32 %70 to <1 x float>
  %72 = extractvalue { i32, i32, i32, i32 } %69, 1
  %73 = bitcast i32 %72 to <1 x float>
  %74 = extractvalue { i32, i32, i32, i32 } %69, 2
  %75 = bitcast i32 %74 to <1 x float>
  %76 = extractvalue { i32, i32, i32, i32 } %69, 3
  %77 = bitcast i32 %76 to <1 x float>
  %78 = extractelement <1 x float> %71, i64 0
  %79 = extractelement <1 x float> %73, i64 0
  %80 = extractelement <1 x float> %75, i64 0
  %81 = extractelement <1 x float> %77, i64 0
  %82 = call { i32, i32, i32, i32 } asm sideeffect "@$4 ld.global.v4.b32 {$0,$1,$2,$3}, [ $5 + 2048];", "=r,=r,=r,=r,b,l"(i1 %47, float addrspace(1)* %59)
  %83 = extractvalue { i32, i32, i32, i32 } %82, 0
  %84 = bitcast i32 %83 to <1 x float>
  %85 = extractvalue { i32, i32, i32, i32 } %82, 1
  %86 = bitcast i32 %85 to <1 x float>
  %87 = extractvalue { i32, i32, i32, i32 } %82, 2
  %88 = bitcast i32 %87 to <1 x float>
  %89 = extractvalue { i32, i32, i32, i32 } %82, 3
  %90 = bitcast i32 %89 to <1 x float>
  %91 = extractelement <1 x float> %84, i64 0
  %92 = extractelement <1 x float> %86, i64 0
  %93 = extractelement <1 x float> %88, i64 0
  %94 = extractelement <1 x float> %90, i64 0
  %95 = getelementptr float, float addrspace(1)* %1, i32 %27
  %96 = getelementptr float, float addrspace(1)* %95, i32 0
  %97 = getelementptr float, float addrspace(1)* %1, i32 %29
  %98 = getelementptr float, float addrspace(1)* %97, i32 1
  %99 = getelementptr float, float addrspace(1)* %1, i32 %31
  %100 = getelementptr float, float addrspace(1)* %99, i32 2
  %101 = getelementptr float, float addrspace(1)* %1, i32 %33
  %102 = getelementptr float, float addrspace(1)* %101, i32 3
  %103 = getelementptr float, float addrspace(1)* %1, i32 %35
  %104 = getelementptr float, float addrspace(1)* %103, i32 512
  %105 = getelementptr float, float addrspace(1)* %1, i32 %37
  %106 = getelementptr float, float addrspace(1)* %105, i32 513
  %107 = getelementptr float, float addrspace(1)* %1, i32 %39
  %108 = getelementptr float, float addrspace(1)* %107, i32 514
  %109 = getelementptr float, float addrspace(1)* %1, i32 %41
  %110 = getelementptr float, float addrspace(1)* %109, i32 515
  %111 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %112 = urem i32 %111, 32
  %113 = call { i32, i32, i32, i32 } asm sideeffect "@$4 ld.global.v4.b32 {$0,$1,$2,$3}, [ $5 + 0];", "=r,=r,=r,=r,b,l"(i1 %43, float addrspace(1)* %95)
  %114 = extractvalue { i32, i32, i32, i32 } %113, 0
  %115 = bitcast i32 %114 to <1 x float>
  %116 = extractvalue { i32, i32, i32, i32 } %113, 1
  %117 = bitcast i32 %116 to <1 x float>
  %118 = extractvalue { i32, i32, i32, i32 } %113, 2
  %119 = bitcast i32 %118 to <1 x float>
  %120 = extractvalue { i32, i32, i32, i32 } %113, 3
  %121 = bitcast i32 %120 to <1 x float>
  %122 = extractelement <1 x float> %115, i64 0
  %123 = extractelement <1 x float> %117, i64 0
  %124 = extractelement <1 x float> %119, i64 0
  %125 = extractelement <1 x float> %121, i64 0
  %126 = call { i32, i32, i32, i32 } asm sideeffect "@$4 ld.global.v4.b32 {$0,$1,$2,$3}, [ $5 + 2048];", "=r,=r,=r,=r,b,l"(i1 %47, float addrspace(1)* %103)
  %127 = extractvalue { i32, i32, i32, i32 } %126, 0
  %128 = bitcast i32 %127 to <1 x float>
  %129 = extractvalue { i32, i32, i32, i32 } %126, 1
  %130 = bitcast i32 %129 to <1 x float>
  %131 = extractvalue { i32, i32, i32, i32 } %126, 2
  %132 = bitcast i32 %131 to <1 x float>
  %133 = extractvalue { i32, i32, i32, i32 } %126, 3
  %134 = bitcast i32 %133 to <1 x float>
  %135 = extractelement <1 x float> %128, i64 0
  %136 = extractelement <1 x float> %130, i64 0
  %137 = extractelement <1 x float> %132, i64 0
  %138 = extractelement <1 x float> %134, i64 0
  %139 = fadd float %78, %122
  %140 = fadd float %79, %123
  %141 = fadd float %80, %124
  %142 = fadd float %81, %125
  %143 = fadd float %91, %135
  %144 = fadd float %92, %136
  %145 = fadd float %93, %137
  %146 = fadd float %94, %138
  %147 = getelementptr float, float addrspace(1)* %2, i32 %27
  %148 = getelementptr float, float addrspace(1)* %147, i32 0
  %149 = getelementptr float, float addrspace(1)* %2, i32 %29
  %150 = getelementptr float, float addrspace(1)* %149, i32 1
  %151 = getelementptr float, float addrspace(1)* %2, i32 %31
  %152 = getelementptr float, float addrspace(1)* %151, i32 2
  %153 = getelementptr float, float addrspace(1)* %2, i32 %33
  %154 = getelementptr float, float addrspace(1)* %153, i32 3
  %155 = getelementptr float, float addrspace(1)* %2, i32 %35
  %156 = getelementptr float, float addrspace(1)* %155, i32 512
  %157 = getelementptr float, float addrspace(1)* %2, i32 %37
  %158 = getelementptr float, float addrspace(1)* %157, i32 513
  %159 = getelementptr float, float addrspace(1)* %2, i32 %39
  %160 = getelementptr float, float addrspace(1)* %159, i32 514
  %161 = getelementptr float, float addrspace(1)* %2, i32 %41
  %162 = getelementptr float, float addrspace(1)* %161, i32 515
  %163 = insertelement <1 x float> undef, float %139, i64 0
  %164 = bitcast <1 x float> %163 to i32
  %165 = insertelement <1 x float> undef, float %140, i64 0
  %166 = bitcast <1 x float> %165 to i32
  %167 = insertelement <1 x float> undef, float %141, i64 0
  %168 = bitcast <1 x float> %167 to i32
  %169 = insertelement <1 x float> undef, float %142, i64 0
  %170 = bitcast <1 x float> %169 to i32
  call void asm sideeffect "@$0 st.global.v4.b32 [ $1 + 0] , {$2,$3,$4,$5};", "b,l,r,r,r,r"(i1 %43, float addrspace(1)* %147, i32 %164, i32 %166, i32 %168, i32 %170)
  %171 = insertelement <1 x float> undef, float %143, i64 0
  %172 = bitcast <1 x float> %171 to i32
  %173 = insertelement <1 x float> undef, float %144, i64 0
  %174 = bitcast <1 x float> %173 to i32
  %175 = insertelement <1 x float> undef, float %145, i64 0
  %176 = bitcast <1 x float> %175 to i32
  %177 = insertelement <1 x float> undef, float %146, i64 0
  %178 = bitcast <1 x float> %177 to i32
  call void asm sideeffect "@$0 st.global.v4.b32 [ $1 + 2048] , {$2,$3,$4,$5};", "b,l,r,r,r,r"(i1 %47, float addrspace(1)* %155, i32 %172, i32 %174, i32 %176, i32 %178)
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0

attributes #0 = { nounwind readnone }

!nvvm.annotations = !{!0, !1}

!0 = !{void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32)* @add_kernel__Pfp32_Pfp32_Pfp32_i32__4c1024, !"kernel", i32 1}
!1 = !{void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32)* @add_kernel__Pfp32_Pfp32_Pfp32_i32__4c1024, !"maxntidx", i32 128}
