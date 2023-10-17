function subset= get_subset(batch, num_pts)
    if num_pts>= size(batch,1)
        subset=batch;
    else
        indices=randi(size(batch,1), [num_pts,1]);
        subset=batch(indices,:);
    end
