import { useState, useEffect } from 'react';
import Error from '../components/Message';
import { toast } from 'react-toastify';

const useTMDb = (id, initialRun = true, subEndpoint = "") => {
    const [loading, setLoading] = useState(true);
    const [data, setData] = useState(null);
    const [run, setRun] = useState(initialRun);

    useEffect(() => {
        if(run) {
            fetchData();
        }
    }, [run]);

    function doRun() {
        setRun(true);
    }

    const fetchData = async () => {
        setLoading(true);
        try {
            let baseUrl = `https://api.themoviedb.org/3/movie/${id}`;
            if (subEndpoint.length > 0) {
                baseUrl += `/${subEndpoint}`;
            }
            const response = await fetch(baseUrl, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJiYjc0MjIxYWU5MDI0OGNiYzg3YjEzNjBiZTRlZTMzZSIsInN1YiI6IjY1ZDRiN2RjMjNkMjc4MDE3Y2Y0ZGU1MCIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.klS2C5qbbUIjT_uoK5RXVRvWERFFJ-OLfPUQvqZakFE'
                },
            });
            const json = await response.json();
            if (!response.ok) throw new Error(json.status_message || "Failed to fetch");
            setData(json);
        } catch (error) {
            toast.error(error.message);
        } finally {
            setLoading(false);
        }
    };

    return { 
        loading, 
        data, 
        reloadData: () => setLoading(true),
        doRun
    };
};

export default useTMDb;
