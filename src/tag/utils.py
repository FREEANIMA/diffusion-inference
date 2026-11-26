def build_tag_maps(tags_type_df):
    s = tags_type_df[["tag", "type", "dup", "count"]].drop_duplicates(subset="tag", keep="first")

    dup_map = dict(zip(s["tag"], s["dup"].fillna(s["tag"])))

    common = s[s["count"].fillna(0) >= 100]
    type_map = dict(zip(common["tag"], common["type"]))

    t = common.assign(dup=common["dup"].fillna(common["tag"]))
    t = t.sort_values(["count", "dup"], ascending=[False, True])
    t = t.drop_duplicates(subset="dup", keep="first")

    tag_tree = {}
    for typ, tag, cnt in zip(t["type"], t["dup"], t["count"]):
        if isinstance(typ, str) and isinstance(tag, str):
            tag_tree.setdefault(typ, []).append([tag, int(cnt)])

    return type_map, dup_map, tag_tree

def unk_separation(dup_map, tags_list) -> tuple[list[list], list[list]]:
    unks_list, dups_list = [], []
    for tags in tags_list:
        unk_list, dup_list = [], []
        for tag in tags:
            canon = dup_map.get(tag)
            if canon is not None:
                dup_list.append(canon)
            else:
                unk_list.append(tag)
        unks_list.append(unk_list)
        dups_list.append(dup_list)
    return unks_list, dups_list

def mapping(generation_list, nsfw_text_list, size_text_list, init_text_list, unks_list, NAS: list, type_map: dict) -> list[str]:
    texts = []
    for g in generation_list:
        parts = []
        for _type in NAS:
            type_tag = [tag for tag in g if type_map.get(tag) == _type]
            if type_tag:
                parts.append(f"{_type}: " + ", ".join(type_tag) + ";")
        texts.append(" ".join(parts))

    others_texts = [
        f"others: {', '.join(unks)};" if unks else ""
        for unks in unks_list
    ]

    results = []
    for init, nsfw, size, text, others in zip(
        init_text_list, nsfw_text_list, size_text_list, texts, others_texts
    ):
        row = " ".join(s.strip() for s in (init, nsfw, size, text, others) if s)
        results.append(row)

    return results