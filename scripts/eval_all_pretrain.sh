# Task 1: NS Val
bash scripts/eval_change.sh \
    "checkpoint/stage1_visual_change_pretrain_NS" \
    "/lby_data01/zhaozy/lby/ChangeRef_Clear" \
    NS_FINAL_CLEAN_val \
    0,1,2,3 \
    2>&1 | tee eval_NS.log

# # Task 2: RS
# bash scripts/eval_change.sh \
#     "checkpoint/stage1_visual_change_pretrain_RS" \
#     "/lby_data01/zhaozy/lby/ChangeRef_Clear" \
#     RS_FINAL_CLEAN_val \
#     0,1,2,3 \
#     2>&1 | tee eval_RS.log

# # Task 3: TV
# bash scripts/eval_change.sh \
#     "checkpoint/stage1_visual_change_pretrain_TV" \
#     "/lby_data01/zhaozy/lby/ChangeRef_Clear" \
#     val_FINAL_CLEAN \
#     0,1,2,3 \
#     2>&1 | tee eval_TV.log