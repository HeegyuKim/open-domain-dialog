def levenshtein(s1, s2, normalize=False):
    if len(s1) < len(s2):
        return levenshtein(s2, s1, normalize)

    if len(s2) == 0:
        if normalize:
            return 1.0
        else:
            return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))

        previous_row = current_row

    if normalize:
        return previous_row[-1] / max(len(s1), len(s2))
    else:
        return previous_row[-1]

def levenshtein_batch(s1, s2):
    lev_dists = [levenshtein(a, b, normalize=True) for a, b in zip(s1, s2)]
    lev_dist_mean = sum(lev_dists) / len(lev_dists)
    return lev_dist_mean