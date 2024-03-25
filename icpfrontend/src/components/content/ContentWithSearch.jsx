import { useState, useEffect } from 'react';
import useFetchData from '../../hooks/useFetchData';
import SearchHeader from '../search/SearchHeader';
import { Link } from 'react-router-dom';
import PropTypes from 'prop-types';
import SearchAndNavigateHeader from '../search/SearchAndNavigateHeader';
/**
 * 
 * Show all the countries from the affiliation table, allows for searching of countries by name
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 * 
 */

function ContentWithSearch(props) {
  const {
    endpoint,
    returnLink,
    filterFunction,
    renderedItem,
    enablePagination = false,
    enableDebouncing = false,
    startParams,
    gridLayout = false
  } = props;

  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [page, setPage] = useState(1);

  const endpointModified = () => {
    let modifiedEndpoint = endpoint;

    if (startParams) {
        modifiedEndpoint += `?${startParams}`;
    }

    if (enablePagination) {
        modifiedEndpoint += (startParams ? '&' : '?') + `page=${page}`;
    }

    return modifiedEndpoint;
  }

  const { data, setEndpoint, reloadData, loading } = useFetchData(endpointModified);

  useEffect(() => {
    if (enableDebouncing) {
      const timer = setTimeout(() => {
        setDebouncedSearch(search);
      }, 300);
      return () => {
        clearTimeout(timer);
      };
    }
  }, [search, enableDebouncing]);

  useEffect(() => {
    if (enablePagination || enableDebouncing) {
      let newEndpoint = `${endpoint}`;
      if (debouncedSearch) {
        setPage(1);
        newEndpoint += `?page=${page}`;
        newEndpoint += `&search=${debouncedSearch}`;
      } else {
        newEndpoint += `?page=${page}`;
      }
      setEndpoint(newEndpoint);
      reloadData();
    }
  }, [endpoint, page, debouncedSearch, enablePagination, enableDebouncing]);

  const endOfData = (!loading && data.length === 0);

  function nextPage() {
    if(!endOfData) {
      setPage((prevPage) => prevPage + 1);
    }
  }

  function backPage() {
    if (page > 1) {
      setPage((prevPage) => prevPage - 1);
    }
  }

  const handleSearchChange = (event) => {
    setSearch(event.target.value);
  };

  const filteredData = filterFunction ? filterFunction(data, search) : defaultFilter(data, search);

  function defaultFilter(data, search) {
    return data ? data.filter(item =>
      item.title.toLowerCase().includes(search.toLowerCase())
    ) : [];
  }

  const renderedItemDefault = (item, index) => (
    <Link key={index} to={`/content/${item.id}`}>
      <div className="content-item secondary" >
      <p className="text-md truncate">{item.title}</p>
      </div>
    </Link>
  );

  function renderItems(data) {
    return data.map((item, index) => (
        renderedItem ? renderedItem(item, index) : renderedItemDefault(item, index)
    ));
  }

  return (
    <div className="flex flex-col w-[50vw] h-[70vh] mx-auto overflow-hidden">
      <div className="p-2">
        <Link to={returnLink}>Go back</Link>
        {enablePagination ? (
          <SearchAndNavigateHeader
            page={page}
            nextPage={nextPage}
            backPage={backPage}
            handleSearchChange={handleSearchChange}
          />
        ) : (
          <SearchHeader handleSearchChange={handleSearchChange} />
        )}
      </div>
      {filteredData.length > 0 && (
        <div className={`flex flex-col gap-2 overflow-y-scroll ${gridLayout ? 'grid grid-cols-2 gap-2' : ''}`}>
          {renderItems(filteredData)}
        </div>
      )}
      {filteredData.length === 0 && (
        <div className="flex flex-col gap-1 overflow-y-scroll">
          <p className="text-center">No results found</p>
        </div>
      )}
    </div>
  );
}

ContentWithSearch.propTypes = {
    endpoint: PropTypes.string.isRequired,
    returnLink: PropTypes.string.isRequired,
    filterFunction: PropTypes.func,
    renderedItem: PropTypes.func,
    enablePagination: PropTypes.bool,
    enableDebouncing: PropTypes.bool,
    startParams: PropTypes.string,
    gridLayout: PropTypes.bool
};

export default ContentWithSearch;
